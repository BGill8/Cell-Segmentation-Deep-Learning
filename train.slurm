#!/bin/bash
#SBATCH --job-name=nuclei_instance_seg
#SBATCH --partition=gpu            # Specify the GPU partition
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --cpus-per-task=4          # Number of CPU cores for data loading
#SBATCH --mem=32G                  # Memory required
#SBATCH --time=12:00:00            # Maximum time limit (hrs:min:sec)
#SBATCH --output=outputs/slurm-%j.out

# ==============================================================================
# OSU HPC SLURM Script for Cell Instance Segmentation
# Replace 'your_onid' with your actual OSU ONID username before running.
# ==============================================================================

# Load necessary modules
module load python/3.10
module load cuda/11.8

# Activate your virtual environment
# Ensure you have run 'python -m venv .venv' and installed requirements first.
source .venv/bin/activate

# Run the training script
# We point to the scratch space to ensure high-speed I/O during training.
python src/train.py \
    --data_path /scratch/your_onid/data-science-bowl-2018/stage1_train \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --num_workers 4 \
    --wandb_project "cell-segmentation-ai535"
