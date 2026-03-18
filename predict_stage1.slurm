#!/bin/bash
#SBATCH --job-name=nuclei_predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/predict-%j.out

# Load necessary modules
module load python/3.10
module load cuda/11.8

# Activate your virtual environment
source .venv/bin/activate

# Use the best model (mAP) by default
CHECKPOINT="checkpoints/best_model_mAP.pth.tar"

# Point directly to the Stage 1 Test data
INPUT_DIR="data/data-science-bowl-2018/stage1_test"
OUTPUT_DIR="outputs/stage1_predictions"

# Create output folder if missing
mkdir -p $OUTPUT_DIR

echo "Running prediction on $INPUT_DIR using $CHECKPOINT"

# Run the predict script
python src/predict.py \
    --checkpoint $CHECKPOINT \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --visualize \
    --save_masks \
    --save_original_size