#### File to compute metrics directly from the saved models in ./save_folder
#### The metrics are computed on the validation set, which is loaded from the indices saved in ./save_folder/valid_indices.pth

import torch
from torch import nn
import torch.nn.functional as F
import os
import torchvision as tv
from torch.utils.data.sampler import SubsetRandomSampler
from models import DenseNet

from Platt_scaling import ModelWithPlatt, _ECELoss




class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
    


def compute_metrics(model, data, growth_rate=12, batch_size=256, depth=40):
    """Compute metrics for all models that are in the folder ./save_folder
    The models are files with names like model_*.pth.

    Takes a pretrained DenseNet-CIFAR100 model, and a validation set
    (parameterized by indices on train set).

    model (str) - path to model file
    data (str) - path to directory where data should be loaded from/downloaded


    """

    

    ###### Load validation indices
    valid_indices_filename = './save_folder/valid_indices.pth'
    if not os.path.exists(valid_indices_filename):
        raise RuntimeError('Cannot find file %s to load' % valid_indices_filename)
    valid_indices = torch.load(valid_indices_filename)

    # Regenerate validation set loader
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    test_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])
    valid_set = tv.datasets.CIFAR100(data, train=True, transform=test_transforms, download=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, pin_memory=True, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(valid_indices))

    ##### Load calibrated model
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    
    block_config = [(depth - 4) // 6 for _ in range(3)]
    fresh_model = DenseNet(
        state_dict=state_dict,
        growth_rate=growth_rate,
        block_config=block_config,
        num_classes=100
    ).cuda()

    # Wrap it with ModelWithPlatt
    model_with_platt = ModelWithPlatt(fresh_model).cuda()

    # Load model state dict which corresponds to the trained model
    model_filename = model
    if not os.path.exists(model_filename):
        raise RuntimeError('Cannot find model %s to load' % model_filename)
    state_dict = torch.load(model_filename)

    # Load the saved state_dict
    model_with_platt.load_state_dict(state_dict)

    # Now model_with_platt is ready to use
    model_with_platt.eval()  

    ##### Compute metrics
    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = _ECELoss().cuda()

    # First: collect all the logits and labels for the validation set
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for input, label in valid_loader:
            input = input.cuda()
            logits = model_with_platt(input)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()


    # Calculate NLL and ECE after temperature scaling
    after_Platt_nll = nll_criterion(logits, labels).item()
    after_Platt_ece = ece_criterion(logits, labels).item()
    print('After Platt_scaling - NLL: %.3f, ECE: %.3f' % (after_Platt_nll, after_Platt_ece), '\n')

    
    








    ##### Compute accuracy