import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np 
from sklearn.isotonic import IsotonicRegression
from save_results import save_in_csv

class ModelWithIsotonic(nn.Module):
    """
    A thin decorator, which wraps a model with Isotonic scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithIsotonic, self).__init__()
        self.model = model
        # one isotonic regression for each class
        self.isotonic_regressors = None
        self.num_classes = None


    def forward(self, input):
        logits = self.model(input)
        return self.Isotonic_scale(logits)

    def Isotonic_scale(self, logits):
        """
        Perform Isotonic scaling on logits
        """
        if self.isotonic_regressors is None:
            return logits
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1).cpu().numpy()
        calibrated_probs = np.zeros_like(probs)
        
        # Isotonic regression for each class probability
        for c in range(self.num_classes):
            calibrated_probs[:, c] = self.isotonic_regressors[c].predict(probs[:, c])
        
        # Convert back to tensor
        calibrated_probs = torch.tensor(calibrated_probs, device=logits.device)
        
        # Normalize to ensure valid probability distribution
        calibrated_probs = calibrated_probs / calibrated_probs.sum(dim=1, keepdim=True)
        
        # Clamp to prevent numerical issues
        calibrated_probs = torch.clamp(calibrated_probs, min=1e-7, max=1-1e-7)
        
        # Convert probabilities to logits
        calibrated_logits = torch.log(calibrated_probs)
        
        return calibrated_logits

    # This function probably should live outside of this class, but whatever
    def set_f(self, valid_loader):
        """
        Tune the function f of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before Isotonic scaling
        before_Isotonic_nll = nll_criterion(logits, labels).item()
        before_Isotonic_ece = ece_criterion(logits, labels).item()
        print('\n Before Isotonic_Scaling - NLL: %.3f, ECE: %.3f' % (before_Isotonic_nll, before_Isotonic_ece))

        # Apply isotonic regression for each class
        probs = F.softmax(logits, dim=1).cpu().numpy()
        labels_onehot = F.one_hot(labels, logits.size(1)).float().cpu().numpy()
        
        self.num_classes = logits.size(1)
        self.isotonic_regressors = []
        
        # For each class, fit isotonic regression between predicted probs and actual labels
        for c in range(self.num_classes):
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(probs[:, c], labels_onehot[:, c])
            self.isotonic_regressors.append(ir)

        # Calculate NLL and ECE after Isotonic scaling
        calibrated_logits = self.Isotonic_scale(logits)
        nll = nll_criterion(calibrated_logits, labels).item()
        ece = ece_criterion(calibrated_logits, labels).item()
        accuracy = (calibrated_logits.argmax(dim=1) == labels).float().mean().item()


        print('After Isotonic_scaling - NLL: %.3f, ECE: %.3f' % (nll, ece), '\n')

        # Save results to CSV
        save_in_csv(accuracy, nll, ece, "Isotonic_scaling")

        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for Isotonic scaling, just a cool metric).

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
