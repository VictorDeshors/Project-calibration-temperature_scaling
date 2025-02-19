import torchvision.datasets as datasets
import os

# Define the directory where you want to save the dataset
data_dir = './data/CIFAR100'

# Create the directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

# Download the CIFAR-10 dataset
cifar100_dataset_train = datasets.CIFAR100(root=data_dir, train=True, download=True)
cifar100_dataset_test = datasets.CIFAR100(root=data_dir, train=False, download=True)

# Print the number of training and test examples
print('Number of training examples:', len(cifar100_dataset_train))
print('Number of test examples:', len(cifar100_dataset_test))

# Print the classes
print('Classes:', cifar100_dataset_train.classes)

# Print the shape of the images in the dataset
image, label = cifar100_dataset_train[0]
print('Image shape:', image.shape)


