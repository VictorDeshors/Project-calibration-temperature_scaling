import fire
import os
import sys
import torch
import torchvision as tv
from torch.utils.data.sampler import SubsetRandomSampler
from models import DenseNet
from Isotonic_scaling import ModelWithIsotonic
import train_new_architectures as new_arch


def demo(data, save, model='densenet', pretrained=True, depth=40, growth_rate=12, batch_size=256):
    """
    Applies Isotonic scaling to a trained model.

    Takes a pretrained CIFAR-100 model, and a validation set
    (parameterized by indices on train set).

    NB: the "save" parameter references a DIRECTORY, not a file.
    In that directory, there should be two files:
    - model.pth (model state dict)
    - valid_indices.pth (a list of indices corresponding to the validation set).

    data (str) - path to directory where data should be loaded from/downloaded
    save (str) - directory with necessary files (see above)
    model (str) - model to use (densenet, lenet, resnet18, resnet34, resnet50, mlp)
    """
    # Rediriger les prints vers un fichier .out
    output_file = os.path.join(save, model, 'model_with_Isotonic.out')
    # Créer le dossier si nécessaire
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Redirection de stdout vers le fichier
    sys.stdout = open(output_file, 'w')

    # Load model state dict
    model_filename = os.path.join(save, model, 'model.pth')
    if not os.path.exists(model_filename):
        raise RuntimeError('Cannot find file %s to load' % model_filename)
    state_dict = torch.load(model_filename)

    # Load validation indices
    valid_indices_filename = os.path.join(save, model, 'valid_indices.pth')
    if not os.path.exists(valid_indices_filename):
        raise RuntimeError('Cannot find file %s to load' % valid_indices_filename)
    valid_indices = torch.load(valid_indices_filename)

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


    ################### MODEL ###################
    if model == 'densenet':
        # densenet configuration
        if (depth - 4) % 3:
            raise Exception('Invalid depth')
        block_config = [(depth - 4) // 6 for _ in range(3)]
        
        # Make model, criterion, and optimizer
        orig_model = DenseNet(
            growth_rate=growth_rate,
            block_config=block_config,
            num_classes=100
        ).cuda()

    elif model == 'lenet':
        orig_model = new_arch.LeNetCIFAR100().cuda()

    elif model == 'resnet18' or model == 'resnet34' or model == 'resnet50':
        orig_model = new_arch.get_resnet_model(
            resnet_type=model, 
            pretrained=pretrained, 
            num_classes=100
        ).cuda()

    elif model == 'vgg16':
        orig_model = new_arch.get_vgg_cifar100(pretrained=pretrained).cuda()
    
    else: #MLP model
        orig_model = new_arch.MLPCIFAR100().cuda()


    orig_model.load_state_dict(state_dict)

    # Now we're going to wrap the model with a decorator that adds temperature scaling
    orig_model = ModelWithIsotonic(orig_model)

    # Tune the model with Isotonic regression, and save the results
    orig_model.set_f(valid_loader)
    model_filename = os.path.join(save, model, 'model_with_Isotonic.pth')
    torch.save(orig_model.state_dict(), model_filename)
    print('Temperature scaled model saved to %s' % model_filename)
    print('Done!')
    
    # Fermer le fichier de sortie pour s'assurer que tout est écrit
    sys.stdout.close()
    # Restaurer stdout à sa valeur par défaut
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    """
    Applies temperature scaling to a trained model.

    Takes a pretrained DenseNet-CIFAR100 model, and a validation set
    (parameterized by indices on train set).
    Applies temperature scaling, and saves a temperature scaled version.

    NB: the "save" parameter references a DIRECTORY, not a file.
    In that directory, there should be two files:
    - model.pth (model state dict)
    - valid_indices.pth (a list of indices corresponding to the validation set).

    --data (str) - path to directory where data should be loaded from/downloaded
    --save (str) - directory with necessary files (see above)
    --model (str) - model to use (densenet, lenet, resnet18, resnet34, resnet50, mlp)
    """
    fire.Fire(demo)