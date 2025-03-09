import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import ResNet18_Weights

# Remplacer ces importations si les packages sont installés correctement
import calibration.stats as stats
import calibration.plotting as plotting
import robustness_metrics as rm

# Chargement des données CIFAR-10 avec la normalisation correcte
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

validation_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=128, shuffle=False)

# Chargement d'un modèle pré-entraîné avec la nouvelle méthode
model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
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

# Calcul de l'Adaptive Calibration Error (ACE)
ace = stats.ace(predictions, onehot_targets)
print(f"Adaptive Calibration Error (ACE): {ace:.4f}")

# Calcul du Test-based Calibration Error (TCE)
tce_metric = rm.metrics.ExpectedCalibrationError()
tce_metric.add_batch(predictions, targets)
tce = tce_metric.result()
print(f"Test-based Calibration Error (TCE): {tce:.4f}")

#Calcul du Brier Score (métrique de calibration + confiance)
def brier_score(predictions, targets):
    return np.mean(np.sum((predictions - targets) ** 2, axis=1))

brier = brier_score(predictions, onehot_targets)
print(f"Brier Score: {brier:.4f}")

# Génération du diagramme de fiabilité
fig, ax = plt.subplots()
plotting.reliability_diagram(predictions, onehot_targets, ax=ax)
plt.title("Reliability Diagram")
plt.show()


# Pour les calculs de ECE, ACE, TCE, et le diagramme de fiabilité, vous devrez implémenter ces fonctions
# ou utiliser une bibliothèque alternative si 'calibration' et 'robustness_metrics' ne sont pas disponibles
