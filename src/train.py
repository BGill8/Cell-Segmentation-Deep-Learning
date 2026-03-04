"""
Main training script.
Initializes the model, dataloaders, loss functions, and optimizer.
Contains the training loop and logs metrics directly to Weights & Biases (WandB).
"""
#Check if an NVIDIA GPU is available, otherwise fallback to CPU

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

from dataset import NucleiDataset, transform
from torch.utils.data import DataLoader
import torch

train_dataset = NucleiDataset(
    root_dir="data/data-science-bowl-2018/stage1_train",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,       # start with 1 because instance masks vary
    shuffle=True,
    num_workers=2
)


#before testing to check:
for images, masks in train_loader:
    print(images.shape)
    print(masks.shape)
    break


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 20
for epoch in range(num_epochs):

    for images, masks in train_loader:

        #this is where you define the training...