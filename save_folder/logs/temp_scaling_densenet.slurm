#!/bin/bash
#SBATCH --job-name=temp_scaling_densenet
#SBATCH --output=/users/eleves-b/2022/victor.deshors/Project_Deep/Project-calibration-temperature_scaling/save_folder/logs/temp_scaling_densenet.out
#SBATCH --error=/users/eleves-b/2022/victor.deshors/Project_Deep/Project-calibration-temperature_scaling/save_folder/logs/temp_scaling_densenet.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=4:00:00

# Run the temperature scaling script
python /users/eleves-b/2022/victor.deshors/Project_Deep/Project-calibration-temperature_scaling/demo_new_architectures.py --data /users/eleves-b/2022/victor.deshors/Project_Deep/Project-calibration-temperature_scaling/data --save /users/eleves-b/2022/victor.deshors/Project_Deep/Project-calibration-temperature_scaling/save_folder --model densenet

echo "Completed temperature scaling for model: densenet"
