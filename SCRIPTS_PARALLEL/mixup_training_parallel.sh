#!/bin/bash
#
# This script launches two SLURM jobs on separate nodes to train
# DenseNet and ResNet models with Mixup regularization

# Create directories for logs
# Create directories for models and logs
mkdir -p mixup_model
mkdir -p mixup_model/densenet
mkdir -p mixup_model/resnet34

# Create the SLURM submission script for DenseNet
cat > slurm_densenet.sh << 'EOL'
#!/bin/bash
#SBATCH --job-name=mixup_densenet
#SBATCH --output=mixup_model/densenet/densenet_%j.out
#SBATCH --error=mixup_model/densenet/densenet_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00


# Run the training script
python mixup_training.py \
  --model_type=densenet \
  --epochs=200 \
  --alpha=1.0 \
  --lr=0.1 \
  --batch_size=128 \
  --depth=40 \
  --growth_rate=12 \

echo "DenseNet training complete!"
EOL

# Create the SLURM submission script for ResNet34
cat > slurm_resnet.sh << 'EOL'
#!/bin/bash
#SBATCH --job-name=mixup_resnet
#SBATCH --output=mixup_model/resnet34/resnet34_%j.out
#SBATCH --error=mixup_model/resnet34/resnet34_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00


# Run the training script
python mixup_training.py \
  --model_type=resnet34 \
  --epochs=200 \
  --alpha=1.0 \
  --lr=0.1 \
  --batch_size=128 \
  --pretrained=False \

echo "ResNet34 training complete!"
EOL

# Make the scripts executable
chmod +x slurm_densenet.sh
chmod +x slurm_resnet.sh

# Submit the jobs to SLURM
echo "Submitting DenseNet job..."
densenet_job_id=$(sbatch slurm_densenet.sh | cut -d' ' -f4)
echo "DenseNet job submitted with ID: $densenet_job_id"

echo "Submitting ResNet job..."
resnet_job_id=$(sbatch slurm_resnet.sh | cut -d' ' -f4)
echo "ResNet job submitted with ID: $resnet_job_id"