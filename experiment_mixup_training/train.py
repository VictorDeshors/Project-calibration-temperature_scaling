### compute ECE for new architectures : LeNet, ResNet, VGG, MLP


"""
Training script adapted from demo in
https://github.com/gpleiss/efficient_densenet_pytorch
"""

import fire
import os
import time
import torch
import torchvision as tv
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.models import ResNet
from torchvision.models import VGG
from torch.functional import F  
import csv

################### Not to be changed ###################
class Meter():
    """
    A little helper class which keeps track of statistics during an epoch.
    """
    def __init__(self, name, cum=False):
        """
        name (str or iterable): name of values for the meter
            If an iterable of size n, updates require a n-Tensor
        cum (bool): is this meter for a cumulative value (e.g. time)
            or for an averaged value (e.g. loss)? - default False
        """
        self.cum = cum
        if type(name) == str:
            name = (name,)
        self.name = name

        self._total = torch.zeros(len(self.name))
        self._last_value = torch.zeros(len(self.name))
        self._count = 0.0

    def update(self, data, n=1):
        """
        Update the meter
        data (Tensor, or float): update value for the meter
            Size of data should match size of ``name'' in the initialized args
        """
        self._count = self._count + n
        if torch.is_tensor(data):
            self._last_value.copy_(data)
        else:
            self._last_value.fill_(data)
        self._total.add_(self._last_value)

    def value(self):
        """
        Returns the value of the meter
        """
        if self.cum:
            return self._total
        else:
            return self._total / self._count

    def __repr__(self):
        return '\t'.join(['%s: %.5f (%.3f)' % (n, lv, v)
            for n, lv, v in zip(self.name, self._last_value, self.value())])
    



    

############ Définition du modèle ResNet adapté pour CIFAR-100 ############
def get_resnet_model(pretrained=False, num_classes=100):
    """
    Charge un modèle ResNet34 de torchvision et l'adapte pour CIFAR-100
    """
    model = tv.models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
    
    # Adaptation pour CIFAR-100 (petites images 32x32)
    # Remplacer la première couche convolutive
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Supprimer la couche de pooling initiale qui réduit trop la dimension pour CIFAR
    model.maxpool = nn.Identity()
    
    # Remplacer la couche de classification finale
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model



def run_epoch(loader, model, criterion, optimizer, epoch=0, n_epochs=0, train=True, device=None):
    """
    Run a single training or evaluation epoch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    time_meter = Meter(name='Time', cum=True)
    loss_meter = Meter(name='Loss', cum=False)
    error_meter = Meter(name='Error', cum=False)

    if train:
        model.train()
        print('Training')
    else:
        model.eval()
        ece_crit = _ECELoss().cuda()
        oe_crit = _OE().cuda()
        logits_list = []
        labels_list = []
        print('Evaluating')

    

    end = time.time()
    for i, (inputs, targets) in enumerate(loader):
        if train:
            model.zero_grad()
            optimizer.zero_grad()

            # Forward pass
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = criterion(output, targets)

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.n_iters = optimizer.n_iters + 1 if hasattr(optimizer, 'n_iters') else 1

        else:
            with torch.no_grad():
                # Forward pass
                inputs = inputs.to(device)
                targets = targets.to(device)
                output = model(inputs)
                loss = criterion(output, targets)
                logits_list.append(output)
                labels_list.append(targets)

                # ece = ece_crit(output, targets)
                # oe = oe_crit(output, targets)

        # Accounting
        _, predictions = torch.topk(output, 1)
        error = 1 - torch.eq(predictions, targets).float().mean()
        batch_time = time.time() - end
        end = time.time()


        # Log errors
        time_meter.update(batch_time)
        loss_meter.update(loss)
        error_meter.update(error)
        # print('  '.join([
        #     '%s: (Epoch %d of %d) [%04d/%04d]' % ('Train' if train else 'Eval',
        #         epoch, n_epochs, i + 1, len(loader)),
        #     str(time_meter),
        #     str(loss_meter),
        #     str(error_meter),
        # ]))

    if not train:
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()

        ece = ece_crit(logits, labels)
        oe = oe_crit(logits, labels)
        accuracy = (logits.argmax(1) == labels).float().mean().item() * 100.0

        return time_meter.value(), loss_meter.value(), error_meter.value(), ece, oe, accuracy

    else : 
        return time_meter.value(), loss_meter.value(), error_meter.value()


def train(data, valid_size=5000, seed=None, n_epochs=200, batch_size=64,
          lr=0.1):
    """
    A function to train on CIFAR-100.

    Args:
        data (str) - path to directory where data should be loaded from/downloaded
            (default $DATA_DIR)
        save (str) - path to save the model to (default /tmp)
        model (str) - name of model to use (default 'densenet')

        valid_size (int) - size of validation set
        seed (int) - manually set the random seed (default None)

        depth (int) - depth of the network (number of convolution layers) (default 40)
        growth_rate (int) - number of features added per DenseNet layer (default 12)
        n_epochs (int) - number of epochs for training (default 300)
        batch_size (int) - size of minibatch (default 256)
        lr (float) - initial learning rate
        wd (float) - weight decay
        momentum (float) - momentum
    """
    # Set device (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if seed is not None:
        torch.manual_seed(seed)


    # Data transforms
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    train_transforms = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])

    # Split training into train and validation - needed for calibration
    train_set = tv.datasets.CIFAR100(data, train=True, transform=train_transforms, download=True)
    valid_set = tv.datasets.CIFAR100(data, train=True, transform=test_transforms, download=False)
    indices = torch.randperm(len(train_set))
    train_indices = indices[:len(indices) - valid_size]
    valid_indices = indices[len(indices) - valid_size:] if valid_size else None

    train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(train_indices))
    valid_loader = torch.utils.data.DataLoader(valid_set, pin_memory=True, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(valid_indices))
    
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    


    ################### MODEL ###################
    criterion = nn.CrossEntropyLoss()
    model_orig = get_resnet_model(
        pretrained=False, 
        num_classes=100
    )

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model_orig = torch.nn.DataParallel(model_orig)
    model_wrapper = model_orig.to(device)
    print(model_wrapper)
    
    # ResNet18/34 bénéficient d'un lr initial plus élevé
    optimizer = optim.SGD(
        model_wrapper.parameters(), 
        lr=0.1, 
        momentum=0.9, 
        weight_decay=5e-4,
        nesterov=True
    )

    # Utilisation de MultiStepLR pour réduire le lr aux epochs 60, 120 et 160
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[60, 120, 160], 
        gamma=0.5  # divise par 2 à chaque étape
    )

    # Quand on sauvegarde le modèle, on inclut aussi les indices de validation
    save_path = "./experiment_mixup_training/save"

    # Crée les répertoires s'ils n'existent pas déjà
    os.makedirs(save_path, exist_ok=True)
    

    # Setup CSV file for metrics logging
    model = "resnet34"

    metrics_path = os.path.join(save_path, f"{model}_metrics.csv")
    with open(metrics_path, 'w', newline='') as csvfile:
        metrics_writer = csv.writer(csvfile)
        metrics_writer.writerow(['epoch', 'ece', 'oe', 'acc'])  # Write headers

    ############### TRAINING ###############
    best_error = 1
    model_name = model 
    print('Training', model_name, 'for', n_epochs, 'epochs')
    for epoch in range(1, n_epochs + 1):
        scheduler.step()
        run_epoch(
            loader=train_loader,
            model=model_wrapper,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            train=True,
            device=device
        )
        valid_results = run_epoch(
            loader=valid_loader,
            model=model_wrapper,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            train=False,
            device=device
        )

        # Determine if model is the best
        _, _, valid_error, ece, oe, acc = valid_results
        if valid_error[0] < best_error:
            best_error = valid_error[0]
            print("Epoch =", epoch, "lr =", lr, 'New best error: %.4f' % best_error)

            # Sauvegarde du modèle et des indices de validation
            torch.save(model_orig.state_dict(), os.path.join(save_path, 'model.pth'))
            torch.save(valid_indices, os.path.join(save_path, 'valid_indices.pth'))

        # Log metrics to CSV file
        with open(metrics_path, 'a', newline='') as csvfile:
            metrics_writer = csv.writer(csvfile)
            metrics_writer.writerow([epoch, ece[0], oe[0], acc])  # Assuming these metrics are tensors or lists


    print('Done!')
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
    

class _OE(nn.Module):
    """ Overconfidence error """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_OE, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        oe = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item()) # mask
            prop_in_bin = in_bin.float().mean() # proportion of samples in the bin
            if prop_in_bin.item() > 0: #only proceed if the bin is not empty
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                oe += torch.clamp(avg_confidence_in_bin - accuracy_in_bin, min=0.0) * avg_confidence_in_bin * prop_in_bin

        return oe


if __name__ == '__main__':
    """
    Train on CIFAR-100

    Args:
        --data (str) - path to directory where data should be loaded from/downloaded
            (default $DATA_DIR)

        --valid_size (int) - size of validation set
        --seed (int) - manually set the random seed (default None)
    """
    fire.Fire(train)


