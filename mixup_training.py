import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
from torch.utils.data import DataLoader
from models import DenseNet  # Load your DenseNet or any other architecture
from calibration_evaluation import CalibrationMetrics, temperature_scaling

# ----------------------------
# MIXUP FUNCTION
# ----------------------------
def mixup_data(x, y, alpha=1.0):
    """Apply Mixup to input x and target y."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for Mixup."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ----------------------------
# TRAINING FUNCTION
# ----------------------------
def train_mixup(model, train_loader, valid_loader, test_loader, epochs=100, alpha=1.0, lr=0.1, device="cuda"):
    """Train the model with Mixup regularization."""
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 90], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            mixed_inputs, y_a, y_b, lam = mixup_data(inputs, labels, alpha)

            optimizer.zero_grad()
            outputs = model(mixed_inputs)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (lam * predicted.eq(y_a).sum().item() + (1 - lam) * predicted.eq(y_b).sum().item())
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total

        # Validation Accuracy
        val_acc = evaluate_accuracy(model, valid_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    print("Training Complete! Loading best model...")
    model.load_state_dict(torch.load("best_model.pth"))

    # Evaluate calibration
    evaluate_calibration(model, test_loader, valid_loader, device=device, n_bins=15)

# ----------------------------
# EVALUATION FUNCTION
# ----------------------------
def evaluate_accuracy(model, loader, device):
    """Evaluate accuracy on a dataset."""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return correct / total

# ----------------------------
# LOAD DATASETS
# ----------------------------
def get_dataloaders(batch_size=128):
    """Prepare CIFAR-100 dataloaders with Mixup augmentation."""
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

    indices = torch.randperm(len(train_dataset))
    valid_size = 5000
    train_indices = indices[:-valid_size]
    valid_indices = indices[-valid_size:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_indices), num_workers=2)
    valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(valid_indices), num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, valid_loader, test_loader

# ----------------------------
# MAIN FUNCTION
# ----------------------------
if __name__ == "__main__":
    train_loader, valid_loader, test_loader = get_dataloaders(batch_size=128)
    model = DenseNet(depth=40, growth_rate=12, num_classes=100)
    
    print("Starting Mixup Training on CIFAR-100...")
    train_mixup(model, train_loader, valid_loader, test_loader, epochs=100, alpha=1.0, lr=0.1, device="cuda")
