# Mixup training on CIFAR100 with ResNet34

- We compare accureacy and score of winning class when there is mixup training and when there is not.

- We plot :
    - accuracy/score of winning class for each epoch on the validation set.
    - Overconfidence error for each epoch on the validation set.

- For all experiments, we use batch normalization, weight decay of 5 ×10−4 and trained the network using SGD with Nesterov momentum, training for 200 epochs with an initial learning rate of 0.1 halved at 2 at 60,120 and 160 epochs. Unless otherwise noted, calibration results are reported for the best performing epoch on the validation set.

- 