#!/bin/bash
#SBATCH --job-name=nuclei_stage2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/stage2-%j.out

# Load necessary modules
module load python/3.10
module load cuda/11.8

# Activate your virtual environment
source .venv/bin/activate

# Use the best model (mAP)
CHECKPOINT="checkpoints/best_model_mAP.pth.tar"
INPUT_DIR="data/data-science-bowl-2018/stage2_test_final"
OUTPUT_DIR="outputs/stage2_predictions"

# Create output folder if missing
mkdir -p $OUTPUT_DIR

echo "Running Stage 2 Inference on $INPUT_DIR using $CHECKPOINT"

# Run the predict script
# Note: Stage 2 has many more images, so we increase the time limit
python src/predict.py \
    --checkpoint $CHECKPOINT \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --visualize \
    --save_masks \
    --save_original_size
