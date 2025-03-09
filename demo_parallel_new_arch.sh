#!/bin/bash

# Configuration paths
DATA_DIR="/users/eleves-b/2022/victor.deshors/Project_Deep/Project-calibration-temperature_scaling/data"
SAVE_BASE_DIR="/users/eleves-b/2022/victor.deshors/Project_Deep/Project-calibration-temperature_scaling/save_folder"
SCRIPT_PATH="/users/eleves-b/2022/victor.deshors/Project_Deep/Project-calibration-temperature_scaling/demo_new_architectures.py"

# Create logs directory
mkdir -p "${SAVE_BASE_DIR}/logs"

# List of models to run temperature scaling on
MODELS=(
  "densenet"
  "lenet"
  "resnet18"
  "resnet34"
  "resnet50"
  "mlp"
)

# Generate and submit job for each model
for model in "${MODELS[@]}"; do
  
  # Set job name and save directory
  job_name="temp_scaling_${model}"
  
  # Determine if the model supports pretrained option
  if [[ "$model" == "resnet18" || "$model" == "resnet34" || "$model" == "resnet50" ]]; then
    # These models support pretrained weights
    for pretrained in "True" "False"; do
      specific_job_name="${job_name}_pretrained_${pretrained}"
      save_dir="${SAVE_BASE_DIR}/${model}_pretrained_${pretrained}"
      
      # Create SLURM submission script
      cat > "${SAVE_BASE_DIR}/logs/${specific_job_name}.slurm" << EOF
#!/bin/bash
#SBATCH --job-name=${specific_job_name}
#SBATCH --output=${SAVE_BASE_DIR}/logs/${specific_job_name}.out
#SBATCH --error=${SAVE_BASE_DIR}/logs/${specific_job_name}.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=4:00:00

# Run the temperature scaling script
python ${SCRIPT_PATH} --data ${DATA_DIR} --save ${save_dir} --model ${model} --pretrained ${pretrained} 

echo "Completed temperature scaling for model: ${model} with pretrained=${pretrained}"
EOF
      
      # Submit the job
      sbatch "${SAVE_BASE_DIR}/logs/${specific_job_name}.slurm"
      echo "Job submitted: ${specific_job_name}"
    done
  else
    # Models without pretrained option
    save_dir="${SAVE_BASE_DIR}/${model}"
    
    # Create SLURM submission script
    cat > "${SAVE_BASE_DIR}/logs/${job_name}.slurm" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${SAVE_BASE_DIR}/logs/${job_name}.out
#SBATCH --error=${SAVE_BASE_DIR}/logs/${job_name}.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Run the temperature scaling script
python ${SCRIPT_PATH} --data ${DATA_DIR} --save ${save_dir} --model ${model} 

echo "Completed temperature scaling for model: ${model}"
EOF
    
    # Submit the job
    sbatch "${SAVE_BASE_DIR}/logs/${job_name}.slurm"
    echo "Job submitted: ${job_name}"
  fi
done

echo "All temperature scaling jobs have been submitted to SLURM"