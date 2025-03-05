import sys
import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import calibration.stats as stats
import calibration.binning as binning
import calibration.lenses as lenses
import matplotlib.pyplot as plt
import robustness_metrics as rm

# Définition directe des fonctions de distance
def l1distance(x, y):
    return np.linalg.norm(np.subtract(x, y), ord=1, axis=-1)

def tvdistance(x, y):
    return np.linalg.norm(np.subtract(x, y), ord=1, axis=-1) / 2

def l2distance(x, y):
    return np.linalg.norm(np.subtract(x, y), ord=2, axis=-1)

# Chargement des données CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

validation_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=128, shuffle=False)

# Chargement d'un modèle pré-entraîné
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# Calcul des prédictions et des cibles
all_predictions = []
all_targets = []

with torch.no_grad():
    for inputs, targets in validation_loader:
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        all_predictions.append(probabilities)
        all_targets.append(targets)

# Conversion en numpy arrays
predictions = torch.cat(all_predictions).numpy()
targets = torch.cat(all_targets).numpy()
onehot_targets = np.eye(predictions.shape[1])[targets]

# Calcul de l'ECE
ece = stats.ece(predictions, onehot_targets)
print(f"Expected Calibration Error (ECE): {ece:.4f}")

# Calcul de la Consistency ECE
consistency_ece_mean, consistency_ece_std = stats.consistency_ece(predictions)
print(f"Consistency ECE Mean: {consistency_ece_mean:.4f}, Std: {consistency_ece_std:.4f}")

# Calcul de l'ECE avec Data Dependent Binning
ece_datadependent_binning = stats.ece(predictions, onehot_targets, binning=binning.DataDependentBinning())
print(f"ECE with Data Dependent Binning: {ece_datadependent_binning:.4f}")

# Calcul de l'ECE en utilisant les prédictions les plus confiantes
ece_max = stats.ece(*lenses.maximum_lens(predictions, onehot_targets))
print(f"ECE using Maximum Confidence Lens: {ece_max:.4f}")

# Calcul du Test-based Calibration Error (TCE)
tce_metric = rm.metrics.ExpectedCalibrationError()
tce_metric.add_batch(predictions, targets)
tce = tce_metric.result()
print(f"Test-based Calibration Error (TCE): {tce:.4f}")

# Calcul du Brier Score (métrique de calibration + confiance)
def brier_score(predictions, targets):
    return np.mean(np.sum((predictions - targets) ** 2, axis=1))

brier = brier_score(predictions, onehot_targets)
print(f"Brier Score: {brier:.4f}")

# Génération du diagramme de fiabilité
def plot_reliability_diagram(predictions, targets, bins=10):
    confidences = np.max(predictions, axis=1)
    accuracies = (np.argmax(predictions, axis=1) == np.argmax(targets, axis=1)).astype(float)

    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_accuracy = np.zeros(bins)
    bin_confidence = np.zeros(bins)
    bin_counts = np.zeros(bins)

    for i in range(bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        bin_counts[i] = np.sum(in_bin)

        if bin_counts[i] > 0:
            bin_accuracy[i] = np.mean(accuracies[in_bin])
            bin_confidence[i] = np.mean(confidences[in_bin])
        else:
            bin_accuracy[i] = 0
            bin_confidence[i] = 0

    plt.figure(figsize=(6,6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=2)
    plt.plot(bin_confidence, bin_accuracy, marker="o", linestyle="-", color="blue", label="Model Reliability")
    plt.xlabel("Confiance Moyenne")
    plt.ylabel("Précision Moyenne")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_reliability_diagram(predictions, onehot_targets)

# Exemple d'utilisation des distances
l1_dist = l1distance(predictions, onehot_targets)
tv_dist = tvdistance(predictions, onehot_targets)
l2_dist = l2distance(predictions, onehot_targets)

print(f"L1 Distance: {np.mean(l1_dist):.4f}")
print(f"TV Distance: {np.mean(tv_dist):.4f}")
print(f"L2 Distance: {np.mean(l2_dist):.4f}")
