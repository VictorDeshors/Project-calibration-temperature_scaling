#!/bin/bash

# Configuration des chemins
DATA_DIR="/users/eleves-b/2022/victor.deshors/Project_Deep/Project-calibration-temperature_scaling/data"
SAVE_BASE_DIR="/users/eleves-b/2022/victor.deshors/Project_Deep/Project-calibration-temperature_scaling/save_folder"
SCRIPT_PATH="/users/eleves-b/2022/victor.deshors/Project_Deep/Project-calibration-temperature_scaling/new_architectures.py"  # Chemin vers le script principal

# Liste des modèles à entraîner
MODELS=(
  "densenet"
  "lenet"
  "resnet18"
  "resnet34"
  "resnet50" 
  "mlp"
)

# Liste des valeurs de pretrained
PRETRAINED_VALUES=("True" "False")

# Taille du jeu de validation
VALID_SIZE=5000

# Graine pour la reproductibilité
SEED=42

# Lancement avec SLURM
# Créer le répertoire de logs
mkdir -p "${SAVE_BASE_DIR}/logs"

for model in "${MODELS[@]}"; do
  # Déterminer si le modèle prend en charge l'option pretrained
  if [[ "$model" == "lenet" || "$model" == "densenet" || "$model" == "mlp" ]]; then
    # Ces modèles n'utilisent pas de poids préentraînés
    job_name="${model}"
    save_dir="${SAVE_BASE_DIR}/${model}"
    
    # Créer le script de soumission SLURM
    cat > "${SAVE_BASE_DIR}/logs/${job_name}.slurm" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${SAVE_BASE_DIR}/logs/${job_name}.out
#SBATCH --error=${SAVE_BASE_DIR}/logs/${job_name}.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=24:00:00

python ${SCRIPT_PATH} --data ${DATA_DIR} --save ${save_dir} --model ${model} --n_epochs 300
EOF
    
    # Soumettre le job
    sbatch "${SAVE_BASE_DIR}/logs/${job_name}.slurm"
    echo "Job soumis: ${job_name}"
    
  else
    # Pour les modèles qui supportent l'option pretrained
    for pretrained in "${PRETRAINED_VALUES[@]}"; do
      job_name="${model}_pretrained_${pretrained}"
      save_dir="${SAVE_BASE_DIR}/${job_name}"
      
      # Créer le script de soumission SLURM
      cat > "${SAVE_BASE_DIR}/logs/${job_name}.slurm" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${SAVE_BASE_DIR}/logs/${job_name}.out
#SBATCH --error=${SAVE_BASE_DIR}/logs/${job_name}.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00

python ${SCRIPT_PATH} --data ${DATA_DIR} --save ${save_dir} --model ${model} --pretrained ${pretrained} --n_epochs 300
EOF
      
      # Soumettre le job
      sbatch "${SAVE_BASE_DIR}/logs/${job_name}.slurm"
      echo "Job soumis: ${job_name}"
    done
  fi
done

echo "Tous les jobs ont été soumis à SLURM"