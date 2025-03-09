#!/bin/bash

# Check if GNU parallel is installed


# Default values for parameters
DATA_DIR="./data"
SAVE_DIR="./saved_models"
BATCH_SIZE=256

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA_DIR="$2"
            shift 2
            ;;
        --save)
            SAVE_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories if they don't exist


# List of models to scale
MODELS=("densenet" "lenet" "resnet18" "resnet34" "resnet50" "mlp")

# Function to run temperature scaling for a model
run_temp_scaling() {
    local model=$1
    local data_dir=$2
    local save_dir=$3
    local batch_size=$4
    
    echo "Starting temperature scaling for model: $model"
    
    # Run the Python script for this model
    python demo_new_architectures.py --data "$data_dir" --save "$save_dir" --model "$model" 
    
    echo "Completed temperature scaling for model: $model"
}

export -f run_temp_scaling

# Run temperature scaling for all models in parallel
echo "Running temperature scaling for all models in parallel..."
parallel -j $(nproc) run_temp_scaling ::: "${MODELS[@]}" ::: "$DATA_DIR" ::: "$SAVE_DIR" ::: "$BATCH_SIZE"

echo "All temperature scaling jobs completed!"