"""
Main training script.
Initializes the model, dataloaders, loss functions, and optimizer.
Contains the training loop and logs metrics directly to Weights & Biases (WandB).
"""
# Check if an NVIDIA GPU is available, otherwise fallback to CPU
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")