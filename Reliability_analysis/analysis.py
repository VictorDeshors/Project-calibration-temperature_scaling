import fire
import os
import time
import torch
import torchvision as tv
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler

import sys
import os
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Now you can import from models
from models import DenseNet

from temperature_scaling import ModelWithTemperature


import torch.nn.functional as F


def analysis(data, save, depth=40, growth_rate=12, batch_size=256):
    """"
    Compute several metrics linked to ECE on CIFAR100 with DenseNet."
    """ 
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model state dict
    model_filename = os.path.join(save, 'model_with_temperature.pth')
    if not os.path.exists(model_filename):
        raise RuntimeError('Cannot find file %s to load' % model_filename)
    state_dict = torch.load(model_filename, map_location=device)

    # Load validation indices
    valid_indices_filename = os.path.join(save, 'valid_indices.pth')
    if not os.path.exists(valid_indices_filename):
        raise RuntimeError('Cannot find file %s to load' % valid_indices_filename)
    valid_indices = torch.load(valid_indices_filename, map_location=device)

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

    # Load original model
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]
    orig_model = DenseNet(
        growth_rate=growth_rate,
        block_config=block_config,
        num_classes=100
    ).to(device)

    # Now we're going to wrap the model with a decorator that adds temperature scaling
    orig_model = ModelWithTemperature(orig_model)
    orig_model = orig_model.to(device)

    # Load the model with temperature
    orig_model.load_state_dict(state_dict)

    # First: collect all the logits and labels for the validation set
    logits_list = []
    labels_list = []
    inputs_list = []
    with torch.no_grad():
        for input, label in valid_loader:
            input = input.to(device)
            label = label.to(device)  # Move labels to CUDA
            logits = orig_model(input)
            logits_list.append(logits)
            labels_list.append(label)
            inputs_list.append(input)
        
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)
        inputs = torch.cat(inputs_list).to(device)

        # Create one-hot encoded labels - assuming 100 classes for CIFAR100
        num_classes = 100
        labels_onehot = torch.zeros(labels.size(0), num_classes, device=device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        
        # If you need numpy arrays, you need to move to CPU first
        logits_np = logits.cpu().numpy()
        labels_np = labels.cpu().numpy()
        labels_onehot_np = labels_onehot.cpu().numpy()
        inputs_np = inputs.cpu().numpy()

    print(f"Logits shape: {logits_np.shape}")
    print(f"Labels shape: {labels_onehot_np.shape}")
    ##### Using package calibration ######
    import calibration.stats as stats
    import calibration.lenses as lenses

    print("packages imported")

    ece = stats.ece(logits_np, labels_onehot_np)  
    print("ece calculated")  
    print("consistency ece calculated")
    ece_max = stats.ece(*lenses.maximum_lens(logits_np, labels_onehot_np))
    print("max ece calculated")

    print(f"ECE: {ece:.3f}")
    print(f"Max ECE: {ece_max:.3f}")

    ##### Comparing with manual computation of ECE ######
    
    nll_criterion = nn.CrossEntropyLoss().to(device)
    ece_criterion = _ECELoss().to(device)

    nll = nll_criterion(logits, labels).item()
    ece = ece_criterion(logits, labels).item()
    accuracy = (logits.argmax(1) == labels).float().mean().item() * 100.0
    print("\n With manual computation")
    print(f"NLL: {nll:.3f}")
    print(f"ECE: {ece:.3f}")
    print(f"Accuracy: {accuracy:.3f}")

    return


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


def main():
    fire.Fire(analysis)


if __name__ == "__main__":
    main()