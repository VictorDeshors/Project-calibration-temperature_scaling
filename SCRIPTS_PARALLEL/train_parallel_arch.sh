#!/bin/bash

# Configuration des chemins
DATA_DIR="/users/eleves-b/2022/victor.deshors/Project_Deep/Project-calibration-temperature_scaling/data"
SAVE_BASE_DIR="/users/eleves-b/2022/victor.deshors/Project_Deep/Project-calibration-temperature_scaling/save_folder"
SCRIPT_PATH="/users/eleves-b/2022/victor.deshors/Project_Deep/Project-calibration-temperature_scaling/new_architectures.py"

# Liste des modèles à entraîner
MODELS=(
  "densenet"
  "lenet"
  "resnet18"
  "resnet34"
  "resnet50"
  "mlp"
  "vgg16"
)

# Taille du jeu de validation
VALID_SIZE=5000

# Graine pour la reproductibilité
SEED=42

# Lancement avec SLURM
# Créer le répertoire de logs
mkdir -p "${SAVE_BASE_DIR}/logs"

for model in "${MODELS[@]}"; do
  # Création d'un job pour chaque modèle (sans distinction pretrained)
  job_name="${model}"
  save_dir="${SAVE_BASE_DIR}"
    
  # Créer le script de soumission SLURM
  cat > "${SAVE_BASE_DIR}/logs/${job_name}.slurm" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${SAVE_BASE_DIR}/logs/${job_name}.out
#SBATCH --error=${SAVE_BASE_DIR}/logs/${job_name}.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=24:00:00

python ${SCRIPT_PATH} --data ${DATA_DIR} --save ${save_dir} --model ${model} --n_epochs 300
EOF
    
  # Soumettre le job
  sbatch "${SAVE_BASE_DIR}/logs/${job_name}.slurm"
  echo "Job soumis: ${job_name}"
done

echo "Tous les jobs ont été soumis à SLURM"