import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision as tv
import numpy as np
import os
import fire
from torch.utils.data import DataLoader
from models import DenseNet  # Load your DenseNet or any other architecture
from calibration_evaluation import CalibrationMetrics, temperature_scaling, evaluate_calibration

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
# MODEL DEFINITIONS
# ----------------------------
def get_densenet_model(depth=40, growth_rate=12, num_classes=100):
    """Create a DenseNet model with specified parameters."""
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]

    model = DenseNet(
        growth_rate=growth_rate,
        block_config=block_config,
        num_classes=num_classes
    )
    
    return model

def get_resnet_model(pretrained=False, num_classes=100):
    """
    Load a ResNet34 model from torchvision and adapt it for CIFAR-100
    """
    model = tv.models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
    
    # Adaptation for CIFAR-100 (32x32 images)
    # Replace first convolutional layer
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove initial pooling layer which reduces dimensions too much for CIFAR
    model.maxpool = nn.Identity()
    
    # Replace final classification layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

# ----------------------------
# TRAINING FUNCTION
# ----------------------------
def train_mixup(model_type, train_loader, valid_loader, test_loader, epochs=200, alpha=1.0, 
                lr=0.1, device="cuda", training=True, model_params=None, save_dir=None):
    """
    Train with Mixup regularization, supporting different model architectures.
    
    Args:
        model_type: String, either 'densenet' or 'resnet34'
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        epochs: Number of training epochs
        alpha: Mixup alpha parameter
        lr: Initial learning rate
        device: Device to train on ('cuda' or 'cpu')
        training: Whether to train or just evaluate
        model_params: Dictionary with model-specific parameters
        save_dir: Directory to save the model (if None, uses current directory)
    """
    import csv
    import os
    
    # Set default model parameters if not provided
    if model_params is None:
        model_params = {}
    

    # Create model-specific subdirectory based on model_type
    if model_type.lower() == 'densenet':
        model_specific_dir = "./mixup_model/densenet"
    elif model_type.lower() == 'resnet34':
        model_specific_dir = "./mixup_model/resnet34"
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'densenet' or 'resnet34'")

    # Create the model-specific directory
    os.makedirs(model_specific_dir, exist_ok=True)

    # Set the model save path to use the model-specific directory
    model_save_path = os.path.join(model_specific_dir, f"{model_type}_mixup_model.pth")

    # Create model based on specified type
    if model_type.lower() == 'densenet':
        depth = model_params.get('depth', 40)
        growth_rate = model_params.get('growth_rate', 12)
        num_classes = model_params.get('num_classes', 100)
        model = get_densenet_model(depth, growth_rate, num_classes)
    elif model_type.lower() == 'resnet34':
        pretrained = model_params.get('pretrained', False)
        num_classes = model_params.get('num_classes', 100)
        model = get_resnet_model(pretrained, num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'densenet' or 'resnet34'")

    # Handle multi-GPU if available
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Create/open CSV file for saving results in the model-specific directory
    csv_file = os.path.join(model_specific_dir, f"save_results_mixup_{model_type}.csv")
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Write header if file doesn't exist
            writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Acc', 'ECE', 'Adaptive ECE', 'OE'])
    
    if training:
        # Configure optimizer based on model type
        if model_type.lower() == 'densenet':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 90], gamma=0.1)
        elif model_type.lower() == 'resnet34':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, 
                                 weight_decay=5e-4, nesterov=True)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.5)
        
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
            train_loss = running_loss / len(train_loader)
            train_acc = correct / total
            
            # Validation Accuracy
            val_acc, ece, adaptive_ece, oe = evaluate_accuracy(model, valid_loader, device)
            
            # Print results
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f} | ECE: {ece:.4f} | Adaptive ECE: {adaptive_ece:.4f} | OE: {oe:.4f}")
            
            # Save results to CSV
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, f"{train_loss:.4f}", f"{train_acc:.4f}", 
                                f"{val_acc:.4f}", f"{ece:.4f}", f"{adaptive_ece:.4f}", f"{oe:.4f}"])
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_save_path)
    
    print(f"Training Complete! Loading {model_type} model... \n")
    model.load_state_dict(torch.load(model_save_path))
    
    # Evaluate calibration
    evaluate_calibration(model, test_loader, valid_loader, device=device, n_bins=15)
    
    return model

# ----------------------------
# EVALUATION FUNCTION
# ----------------------------
def evaluate_accuracy(model, loader, device):
    """Evaluate accuracy, ECE, and Adaptive ECE on a dataset."""
    model.eval()
    correct, total = 0, 0
    
    all_probs, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs)
            all_labels.append(labels)

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    
    ece = CalibrationMetrics.compute_ece(all_probs, all_labels, n_bins=15, device=device)
    adaptive_ece = CalibrationMetrics.compute_adaptive_ece(all_probs, all_labels, n_bins=15, device=device)
    oe = CalibrationMetrics.compute_oe(all_probs, all_labels, n_bins=15, device=device)

    return correct / total, ece, adaptive_ece, oe

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
# CLI FUNCTIONS WITH FIRE
# ----------------------------
def train_model(model_type="densenet", epochs=200, alpha=1.0, lr=0.1, batch_size=128, pretrained=False, 
               depth=40, growth_rate=12, training=True):
    """
    Train a model with Mixup regularization using command line arguments.
    
    Args:
        model_type (str): Type of model to train ('densenet' or 'resnet34')
        epochs (int): Number of training epochs
        alpha (float): Mixup alpha parameter
        lr (float): Initial learning rate
        batch_size (int): Batch size for training
        pretrained (bool): Whether to use pretrained model (for ResNet)
        depth (int): DenseNet depth (for DenseNet)
        growth_rate (int): DenseNet growth rate (for DenseNet)
        training (bool): Whether to train or just evaluate
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, valid_loader, test_loader = get_dataloaders(batch_size=batch_size)
    
    if model_type.lower() == "densenet":
        # DenseNet parameters
        model_params = {
            'depth': depth,
            'growth_rate': growth_rate,
            'num_classes': 100
        }
        save_dir = "./mixup_model_densenet/"
        
        print(f"Starting Mixup Training on CIFAR-100 with DenseNet (depth={depth}, growth_rate={growth_rate})...")
        
    elif model_type.lower() == "resnet34":
        # ResNet parameters
        model_params = {
            'pretrained': pretrained,
            'num_classes': 100
        }
        save_dir = "./mixup_model_resnet34/"
        
        print(f"Starting Mixup Training on CIFAR-100 with ResNet34 (pretrained={pretrained})...")
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'densenet' or 'resnet34'")
    
    # Train model
    return train_mixup(
        model_type=model_type,
        train_loader=train_loader, 
        valid_loader=valid_loader, 
        test_loader=test_loader, 
        epochs=epochs, 
        alpha=alpha, 
        lr=lr, 
        device=device, 
        training=training,
        model_params=model_params,
        save_dir=save_dir
    )

# ----------------------------
# MAIN FUNCTION
# ----------------------------
if __name__ == "__main__":
    # Use the Fire library to create a command-line interface automatically
    fire.Fire(train_model)