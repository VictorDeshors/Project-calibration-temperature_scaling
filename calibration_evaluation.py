import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CalibrationMetrics:
    """Classe pour calculer diverses métriques de calibration (ECE, Adaptive ECE, OE)"""
    
    @staticmethod
    def compute_ece(probs, labels, n_bins=10, device=None):
        """
        Compute the Expected Calibration Error (ECE)
        
        Args:
            probs (Tensor): Predicted probabilities (batch_size, num_classes)
            labels (Tensor): Ground-truth labels (batch_size,)
            n_bins (int): Number of bins to use for calibration
            device: Device to place tensors on
            
        Returns:
            ece (float): Expected Calibration Error
        """
        if device is None:
            device = probs.device
            
        confidences, predictions = torch.max(probs, dim=1)  
        accuracies = predictions.eq(labels)

        bin_boundaries = torch.linspace(0, 1, n_bins + 1).to(device)
        ece = torch.tensor(0.0, device=device)

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
    def compute_adaptive_ece(probs, labels, n_bins=10, device=None):
        """
        Compute Adaptive Expected Calibration Error (ECE) with dynamic bins.
        
        Args:
            probs (Tensor): Predicted probabilities (batch_size, num_classes)
            labels (Tensor): Ground-truth labels (batch_size,)
            n_bins (int): Number of bins to use for calibration
            device: Device to place tensors on
            
        Returns:
            ece (float): Adaptive Expected Calibration Error
        """
        if device is None:
            device = probs.device
            
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
    
    @staticmethod
    def compute_oe(probs, labels, n_bins=10, device=None):
        """
        Compute the Overconfidence Error (OE)
        
        Args:
            probs (Tensor): Predicted probabilities (batch_size, num_classes)
            labels (Tensor): Ground-truth labels (batch_size,)
            n_bins (int): Number of bins to use for calibration
            device: Device to place tensors on
            
        Returns:
            oe (float): Overconfidence Error
        """
        if device is None:
            device = probs.device
            
        confidences, predictions = torch.max(probs, dim=1)  
        accuracies = predictions.eq(labels)
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1).to(device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        oe = torch.tensor(0.0, device=device)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculate mask for samples in the current bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            # Calculate proportion of samples in the bin
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:  # Only proceed if the bin is not empty
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                # Only accumulate OE if confidence > accuracy (overconfident)
                oe += torch.clamp(avg_confidence_in_bin - accuracy_in_bin, min=0.0) * avg_confidence_in_bin * prop_in_bin
        
        return oe.item()

class TemperatureScaler(nn.Module):
    """
    Applies Temperature Scaling to a model's logits.
    """
    def __init__(self, device=None):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)  # Initial temperature = 1.0
        if device:
            self.to(device)

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

    temperature_model = TemperatureScaler(device=device)
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
    Evaluate model calibration using ECE, Adaptive ECE, OE and Temperature Scaling.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test set
        valid_loader: DataLoader for validation set (optional, for temperature scaling)
        device: 'cuda' or 'cpu'
        n_bins: Number of bins for calibration metrics
    
    Prints:
        - ECE, Adaptive ECE, and OE before calibration
        - Optimized temperature
        - ECE, Adaptive ECE, and OE after calibration
    """
    model.eval()
    model = model.to(device)
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

    # Compute calibration metrics before temperature scaling
    ece = CalibrationMetrics.compute_ece(all_probs, all_labels, n_bins, device=device)
    adaptive_ece = CalibrationMetrics.compute_adaptive_ece(all_probs, all_labels, n_bins, device=device)
    oe = CalibrationMetrics.compute_oe(all_probs, all_labels, n_bins, device=device)
    
    print(f"ECE before calibration: {ece:.4f}")
    print(f"Adaptive ECE before calibration: {adaptive_ece:.4f}")
    print(f"Overconfidence Error before calibration: {oe:.4f}")

    if valid_loader:
        # Apply Temperature Scaling
        best_temp = temperature_scaling(model, valid_loader, device)
        print(f"Optimized Temperature: {best_temp:.4f}")

        # Recompute probabilities with temperature scaling
        model.eval()
        all_logits = []
        
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                logits = model(inputs)
                all_logits.append(logits)
                
        all_logits = torch.cat(all_logits)
        all_probs_scaled = torch.nn.functional.softmax(all_logits / best_temp, dim=1)

        # Compute calibration metrics after temperature scaling
        ece_after_scaling = CalibrationMetrics.compute_ece(all_probs_scaled, all_labels, n_bins, device=device)
        adaptive_ece_after_scaling = CalibrationMetrics.compute_adaptive_ece(all_probs_scaled, all_labels, n_bins, device=device)
        oe_after_scaling = CalibrationMetrics.compute_oe(all_probs_scaled, all_labels, n_bins, device=device)
        
        print(f"ECE after Temperature Scaling: {ece_after_scaling:.4f}")
        print(f"Adaptive ECE after Temperature Scaling: {adaptive_ece_after_scaling:.4f}")
        print(f"Overconfidence Error after Temperature Scaling: {oe_after_scaling:.4f}")

if __name__ == "__main__":
    print("Calibration Evaluation Script Loaded. Use `evaluate_calibration(model, test_loader, valid_loader, device='cuda')` to test.")



"""
from calibration_evaluation import evaluate_calibration

# Supposons que `model`, `test_loader`, et `valid_loader` existent déjà
# Définir device en fonction de la disponibilité de CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
evaluate_calibration(model, test_loader, valid_loader, device=device, n_bins=15)

A ajouter au script d'entrainement
"""