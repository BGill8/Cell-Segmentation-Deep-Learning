#!/bin/bash
#SBATCH --job-name=image_pred
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/imagePred-%j.out

module load python/3.10
module load cuda/11.8

source .venv/bin/activate

python src/imagePred.py