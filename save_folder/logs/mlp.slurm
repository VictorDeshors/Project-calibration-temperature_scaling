#!/bin/bash
#SBATCH --job-name=mlp
#SBATCH --output=/users/eleves-b/2022/victor.deshors/Project_Deep/Project-calibration-temperature_scaling/save_folder/logs/mlp.out
#SBATCH --error=/users/eleves-b/2022/victor.deshors/Project_Deep/Project-calibration-temperature_scaling/save_folder/logs/mlp.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=24:00:00

python /users/eleves-b/2022/victor.deshors/Project_Deep/Project-calibration-temperature_scaling/new_architectures.py --data /users/eleves-b/2022/victor.deshors/Project_Deep/Project-calibration-temperature_scaling/data --save /users/eleves-b/2022/victor.deshors/Project_Deep/Project-calibration-temperature_scaling/save_folder --model mlp --n_epochs 300
