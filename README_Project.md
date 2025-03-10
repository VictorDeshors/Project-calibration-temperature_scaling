# Calibration Strategies for Deep Learning Models

## Project Overview

This project explores and implements various calibration strategies for deep learning models. It focuses on improving the reliability and accuracy of model predictions using different datasets (CIFAR-10, CIFAR-100, ImageNet) and investigates various metrics such as ECE (Expected Calibration Error), robustness, and overconfidence patterns.

## Project Structure

.
├── Isotonic_scaling.py
├── Platt_scaling.py
├── README.md
├── SCRIPTS_PARALLEL/
│   ├── demo_parallel_new_architectures.sh
│   ├── mixup_training_parallel.sh
│   └── train_parallel_arch.sh
├── calibration_evaluation.py
├── demo.py
├── experiment_mixup_training/
│   ├── README.md
│   └── train.py
├── mixup_model/
│   ├── densenet/
│   └── resnet34/
├── mixup_training.py
├── models/
│   ├── densenet/
│   ├── lenet/
│   ├── mlp/
│   ├── resnet18/
│   └── vgg16/
├── reliability_analysis/
│   ├── README.md
│   └── analysis.py
├── requirements.txt
├── save_folder/
├── save_results.py
├── setup.cfg
└── train_new_architectures.py

## Key Components

1. **Main Python Scripts:**
   - `Isotonic_scaling.py`: Implements isotonic regression for calibration.
   - `Platt_scaling.py`: Implements Platt scaling for calibration.
   - `calibration_evaluation.py`: Evaluates the calibration of models.
   - `demo.py`: Demonstrates the usage of various components.
   - `train.py`: Main script for training DenseNet on CIFAR100.
   - `mixup_training.py`: Implements mixup training strategy.
   - `save_results.py`: Saves experimental results in a csv.
   - `train_new_architectures.py`: Trains several models on CIFAR100.

2. **SCRIPTS_PARALLEL:**
   Contains scripts for parallel training and evaluation of models.

3. **experiment_mixup_training:**
   Dedicated to experiments with mixup training strategy.

4. **mixup_model and models:**
   Contain model-specific files, including trained models, error logs, and outputs.

5. **reliability_analysis:**
   Contains scripts for analyzing the reliability of model predictions.

6. **save_folder:**
   Stores saved models, results, and validation indices.

## Getting Started

1. Clone the repository:
   ```
   git clone <repository-url>
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Choose an experiment or component to work on (e.g., temperature scaling, mixup training).

## Key Features

- Implementation of various calibration strategies (Temperature Scaling, Platt Scaling, Isotonic Regression)
- Mixup training for improved calibration
- Support for multiple architectures (ResNet, DenseNet, VGG, LeNet, MLP)
- Parallel training scripts for efficiency
- Comprehensive reliability analysis tools