"""
Inference and post-processing script for the adapted U-Net.
Applies the Marker-Controlled Watershed algorithm to the model's predicted 
distance maps and probability outputs. This algorithmic step converts the 
continuous deep learning outputs into discrete, uniquely labeled cell instances.
"""
# Check if an NVIDIA GPU is available, otherwise fallback to CPU
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")