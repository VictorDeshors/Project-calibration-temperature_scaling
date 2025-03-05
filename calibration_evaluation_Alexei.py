import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CalibrationMetrics:
    """Classe pour calculer l'Expected Calibration Error (ECE)"""
    
    @staticmethod
    def compute_ece(probs, labels, n_bins=10):
        """
        Compute the Expected Calibration Error (ECE)
        
        Args:
            probs (Tensor): Predicted probabilities (batch_size, num_classes)
            labels (Tensor): Ground-truth labels (batch_size,)
            n_bins (int): Number of bins to use for calibration
            
        Returns:
            ece (float): Expected Calibration Error
        """
        confidences, predictions = torch.max(probs, dim=1)  
        accuracies = predictions.eq(labels)

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = torch.tensor(0.0)

        for i in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.any():
                bin_acc = accuracies[in_bin].float().mean()
                bin_conf = confidences[in_bin].mean()
                bin_weight = in_bin.float().mean()
                ece += bin_weight * torch.abs(bin_conf - bin_acc)

        return ece.item()

    @staticmethod
    def compute_adaptive_ece(probs, labels, n_bins=10):
        """
        Compute Adaptive Expected Calibration Error (ECE) with dynamic bins.
        """
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels)

        sorted_confidences, indices = torch.sort(confidences)
        sorted_accuracies = accuracies[indices]

        bin_size = len(confidences) // n_bins
        ece = 0.0

        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < n_bins - 1 else len(confidences)
            
            bin_conf = sorted_confidences[start:end].mean()
            bin_acc = sorted_accuracies[start:end].float().mean()
            bin_weight = (end - start) / len(confidences)
            
            ece += bin_weight * abs(bin_conf - bin_acc)

        return ece

class TemperatureScaler(nn.Module):
    """
    Applies Temperature Scaling to a model's logits.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)  # Initial temperature = 1.0

    def forward(self, logits):
        return logits / self.temperature  # Scale logits

def temperature_scaling(model, valid_loader, device="cuda"):
    """
    Optimize temperature on a validation set to improve calibration.
    
    Args:
        model: Trained model
        valid_loader: Validation data loader
        device: 'cuda' or 'cpu'
    
    Returns:
        best_temperature (float): Optimized temperature value
    """
    model.eval()
    logits_list, labels_list = [], []

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            logits_list.append(logits)
            labels_list.append(labels)

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    temperature_model = TemperatureScaler().to(device)
    optimizer = optim.LBFGS([temperature_model.temperature], lr=0.01, max_iter=50)

    def loss_fn():
        optimizer.zero_grad()
        scaled_logits = temperature_model(logits)
        loss = nn.CrossEntropyLoss()(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(loss_fn)
    
    return temperature_model.temperature.item()

def evaluate_calibration(model, test_loader, valid_loader=None, device="cuda", n_bins=15):
    """
    Evaluate model calibration using ECE and Temperature Scaling.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test set
        valid_loader: DataLoader for validation set (optional, for temperature scaling)
        device: 'cuda' or 'cpu'
        n_bins: Number of bins for ECE
    
    Prints:
        - ECE before and after calibration
        - Optimized temperature
    """
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            probs = torch.nn.functional.softmax(logits, dim=1)
            all_probs.append(probs)
            all_labels.append(labels)

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    # Compute ECE before calibration
    ece = CalibrationMetrics.compute_ece(all_probs, all_labels, n_bins)
    adaptive_ece = CalibrationMetrics.compute_adaptive_ece(all_probs, all_labels, n_bins)
    
    print(f"ECE before calibration: {ece:.4f}")
    print(f"Adaptive ECE before calibration: {adaptive_ece:.4f}")

    if valid_loader:
        # Apply Temperature Scaling
        best_temp = temperature_scaling(model, valid_loader, device)
        print(f"Optimized Temperature: {best_temp:.4f}")

        # Recompute probabilities with temperature scaling
        model.eval()
        all_logits = torch.cat([model(inputs.to(device)) for inputs, _ in test_loader])
        all_probs_scaled = torch.nn.functional.softmax(all_logits / best_temp, dim=1)

        # Compute ECE after calibration
        ece_after_scaling = CalibrationMetrics.compute_ece(all_probs_scaled, all_labels, n_bins)
        print(f"ECE after Temperature Scaling: {ece_after_scaling:.4f}")

if __name__ == "__main__":
    print("Calibration Evaluation Script Loaded. Use `evaluate_calibration(model, test_loader, valid_loader)` to test.")



"""
from calibration_evaluation import evaluate_calibration

# Supposons que `model`, `test_loader`, et `valid_loader` existent déjà
evaluate_calibration(model, test_loader, valid_loader, device="cuda", n_bins=15)

A ajouter au script d'entrainement
"""