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
