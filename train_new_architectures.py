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
from models import DenseNet
from torchvision.models import ResNet
from torchvision.models import VGG
# from torchvision.models import MLP

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
    


############ Définition du modèle LeNet adapté pour CIFAR-100 ############
class LeNetCIFAR100(nn.Module):
    def __init__(self, num_classes=100):
        super(LeNetCIFAR100, self).__init__()
        # Adaptation de LeNet pour les images RGB 32x32 de CIFAR-100
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3 canaux RGB au lieu de 1 canal pour MNIST
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # Calcul des dimensions après les couches convolutives et de pooling
        # CIFAR-100: 32x32 -> conv1 -> 28x28 -> pool -> 14x14 -> conv2 -> 10x10 -> pool -> 5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)  # 100 classes pour CIFAR-100
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

############ Définition du modèle MLP adapté pour CIFAR-100 ############
class MLPCIFAR100(nn.Module):
    def __init__(self, hidden_size=1024):
        super(MLPCIFAR100, self).__init__()
        # CIFAR100: 3 canaux x 32x32 pixels = 3072 entrées
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(3 * 32 * 32, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 100)  # 100 classes pour CIFAR100
        )
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x
    

############ Définition du modèle ResNet adapté pour CIFAR-100 ############
def get_resnet_model(resnet_type="resnet18", pretrained=True, num_classes=100):
    """
    Charge un modèle ResNet de torchvision et l'adapte pour CIFAR-100
    """
    # Sélection du modèle ResNet
    if resnet_type == "resnet18":
        model = tv.models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    elif resnet_type == "resnet34":
        model = tv.models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
    elif resnet_type == "resnet50":
        model = tv.models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
    else:
        raise ValueError(f"Type de ResNet non supporté: {resnet_type}")
    
    # Adaptation pour CIFAR-100 (petites images 32x32)
    # Remplacer la première couche convolutive
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Supprimer la couche de pooling initiale qui réduit trop la dimension pour CIFAR
    model.maxpool = nn.Identity()
    
    # Remplacer la couche de classification finale
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

# ############ Définition du modèle VGG adapté pour CIFAR-100 ############
def get_vgg_cifar100(pretrained=False):
    """
    Crée un modèle VGG adapté pour le dataset CIFAR100.
    
    Args:
        pretrained (bool): Si True, utilise les poids pré-entraînés sur ImageNet
                          Sinon, initialise avec des poids aléatoires
                       
    Returns:
        Un modèle VGG adapté pour CIFAR100
    """
    # Charger le modèle VGG16 (avec ou sans pré-entraînement)
    if pretrained:
        vgg = tv.models.vgg16(weights=tv.models.VGG16_Weights.DEFAULT)
    else:
        vgg = tv.models.vgg16(weights=None)
    
    # Calculer la taille de sortie après les couches convolutives
    # Pour des images 32x32 passées dans VGG16:
    # Après 5 niveaux de maxpool (chacun divisant la taille par 2)
    # La taille spatiale sera: 32 ÷ (2^5) = 32 ÷ 32 = 1x1
    # Avec 512 canaux, la taille totale sera: 512 × 1 × 1 = 512
    
    # Créer une classe personnalisée qui va adapter VGG16 pour CIFAR100
    class VGG16CIFAR100(nn.Module):
        def __init__(self, original_vgg):
            super(VGG16CIFAR100, self).__init__()
            # Conserver les couches convolutives
            self.features = original_vgg.features
            
            # Remplacer le classificateur pour CIFAR100
            self.classifier = nn.Sequential(
                nn.Linear(512, 512),  # Pour des images 32x32, la sortie des features est 512
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(512, 100)  # 100 classes pour CIFAR100
            )
            
            # Initialiser les poids du nouveau classificateur
            for m in self.classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            # Passer à travers les couches convolutives
            x = self.features(x)
            # Aplatir le tenseur
            x = torch.flatten(x, 1)
            # Passer à travers le classificateur
            x = self.classifier(x)
            return x
    
    # Créer et retourner le modèle adapté
    return VGG16CIFAR100(vgg)







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

    return time_meter.value(), loss_meter.value(), error_meter.value()


def train(data, save, model='densenet', pretrained=True, valid_size=5000, seed=None,
          depth=40, growth_rate=12, n_epochs=300, batch_size=64,
          lr=0.1, wd=0.0001, momentum=0.9):
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

    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)


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
    # IMPORTANT! We need to use the same validation set for temperature
    # scaling, so we're going to save the indices for later
    train_set = tv.datasets.CIFAR100(data, train=True, transform=train_transforms, download=True)
    valid_set = tv.datasets.CIFAR100(data, train=True, transform=test_transforms, download=False)
    indices = torch.randperm(len(train_set))
    train_indices = indices[:len(indices) - valid_size]
    valid_indices = indices[len(indices) - valid_size:] if valid_size else None

    # Make dataloaders
    train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(train_indices))
    valid_loader = torch.utils.data.DataLoader(valid_set, pin_memory=True, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(valid_indices))
    
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    


    ################### MODEL ###################
    criterion = nn.CrossEntropyLoss()

    if model == 'densenet':
        # Get densenet configuration
        if (depth - 4) % 3:
            raise Exception('Invalid depth')
        block_config = [(depth - 4) // 6 for _ in range(3)]
        
        # Make model_orig, criterion, and optimizer
        model_orig = DenseNet(
            growth_rate=growth_rate,
            block_config=block_config,
            num_classes=100
        )
        
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            model_orig = torch.nn.DataParallel(model_orig)
        model_wrapper = model_orig.to(device)
        print(model_wrapper)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model_wrapper.parameters(), lr=lr, momentum=momentum, nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)

    elif model == 'lenet':
        model_orig = LeNetCIFAR100()

        if device.type == "cuda" and torch.cuda.device_count() > 1:
            model_orig = torch.nn.DataParallel(model_orig)
        model_wrapper = model_orig.to(device)
        print(model_wrapper)
        
        # LeNet est plus simple, peut utiliser un lr plus élevé
        optimizer = optim.SGD(
            model_wrapper.parameters(), 
            lr=0.01, 
            momentum=0.9, 
            weight_decay=5e-4,
            nesterov=True
        )
        # MultiStepLR fonctionne bien pour les architectures plus simples
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[int(0.5 * n_epochs), int(0.75 * n_epochs)], 
            gamma=0.1
        )

    elif model == 'resnet18' or model == 'resnet34':

        model_orig = get_resnet_model(
            resnet_type=model, 
            pretrained=pretrained, 
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
        # Cosine scheduler fonctionne bien avec ResNet
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    elif model == 'resnet50':
        model_orig = get_resnet_model(
            resnet_type="resnet50", 
            pretrained=pretrained, 
            num_classes=100
        )

        if device.type == "cuda" and torch.cuda.device_count() > 1:
            model_orig = torch.nn.DataParallel(model_orig)
        model_wrapper = model_orig.to(device)
        print(model_wrapper)
        
        # Pour les modèles plus profonds comme ResNet50
        optimizer = optim.SGD(
            model_wrapper.parameters(), 
            lr=0.1, 
            momentum=0.9, 
            weight_decay=1e-4,
            nesterov=True
        )
        # Cosine scheduler pour les modèles profonds
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    elif model == 'vgg16':
        model_orig = get_vgg_cifar100(pretrained=pretrained)

        if device.type == "cuda" and torch.cuda.device_count() > 1:
            model_orig = torch.nn.DataParallel(model_orig)
        model_wrapper = model_orig.to(device)
        print(model_wrapper)
        
        # VGG bénéficie d'un learning rate plus faible et plus de régularisation
        optimizer = optim.SGD(
            model_wrapper.parameters(), 
            lr=0.01, 
            momentum=0.9, 
            weight_decay=5e-4,
            nesterov=True
        )
        # VGG marche bien avec un scheduler qui réduit progressivement le lr
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    else: #MLP model
        model_orig = MLPCIFAR100()

        if device.type == "cuda" and torch.cuda.device_count() > 1:
            model_orig = torch.nn.DataParallel(model_orig)
        model_wrapper = model_orig.to(device)
        print(model_wrapper)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model_wrapper.parameters(), lr=lr, momentum=momentum, nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)


    




    ############### TRAINING ###############
    best_error = 1
    model_name = model + ('_pretrained' if pretrained else '_random')
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
        _, _, valid_error = valid_results
        if valid_error[0] < best_error:
            best_error = valid_error[0]
            print("Epoch =", epoch, "lr =", lr, 'New best error: %.4f' % best_error)


            # Quand on sauvegarde le modèle, on inclut aussi les indices de validation
            save_path = os.path.join(save, model)

            # Crée les répertoires s'ils n'existent pas déjà
            os.makedirs(save_path, exist_ok=True)

            # Sauvegarde du modèle et des indices de validation
            torch.save(model_orig.state_dict(), os.path.join(save_path, 'model.pth'))
            torch.save(valid_indices, os.path.join(save_path, 'valid_indices.pth'))


    print('Done!')


if __name__ == '__main__':
    """
    Train on CIFAR-100

    Args:
        --data (str) - path to directory where data should be loaded from/downloaded
            (default $DATA_DIR)
        --save (str) - path to save the model to (default /tmp)
        --model (str) - name of model to use (default 'densenet', others : 'lenet', 'resnet18', 'resnet34', 'resnet50', 'vgg16')
        --pretrained (bool) - use a pretrained model (default False)

        --valid_size (int) - size of validation set
        --seed (int) - manually set the random seed (default None)
    """
    fire.Fire(train)

