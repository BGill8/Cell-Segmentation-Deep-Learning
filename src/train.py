"""
Main training script for Cell Instance Segmentation.
Features:
- Multi-task learning (Semantic Mask + Distance Map)
- HPC-ready with argparse and SLURM integration
- Automatic validation split from stage1_train
- mAP calculation using Watershed post-processing
- Logging to Weights & Biases (WandB)
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb
import numpy as np

# Local imports
from dataset import NucleiDataset, transform, val_transform
from model import UNetInstanceSeg
from loss import MultiTaskLoss
from evaluate import run_inference
from metrics import mean_average_precision
from utils import save_checkpoint, visualize_prediction

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    metrics_sum = {"bce": 0, "dice": 0, "mse": 0}
    
    pbar = tqdm(loader, desc="Training")
    for images, targets, _ in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(images)
        loss, loss_dict = criterion(predictions, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        for k in metrics_sum:
            metrics_sum[k] += loss_dict[k]
            
        pbar.set_postfix(loss=loss.item())
        
    avg_loss = total_loss / len(loader)
    return avg_loss, {k: v / len(loader) for k, v in metrics_sum.items()}

def validate(model, loader, device, epoch=None):
    model.eval()
    all_mAP = []
    
    # We only visualize the first sample of the validation set
    visualized = False
    
    with torch.no_grad():
        for images, targets, labeled_masks in tqdm(loader, desc="Validating"):
            images = images.to(device)
            
            # targets: [B, 2, H, W]
            # labeled_masks: [B, H, W]
            
            for i in range(images.size(0)):
                img = images[i]
                true_labeled = labeled_masks[i].cpu().numpy()
                
                # Run inference (includes model forward + watershed)
                pred_labeled = run_inference(model, img)
                
                # Calculate mAP for this image
                mAP = mean_average_precision(pred_labeled, true_labeled)
                all_mAP.append(mAP)
                
                # Visualize the first image of the batch to WandB
                if not visualized:
                    # Get model raw outputs for visualization
                    output = model(img.unsqueeze(0))
                    pred_semantic = torch.sigmoid(output[0, 0]).cpu().numpy()
                    pred_dist = output[0, 1].cpu().numpy()
                    
                    visualize_prediction(
                        img, 
                        (targets[i, 0] > 0.5).cpu().numpy(), 
                        pred_semantic, 
                        pred_dist, 
                        pred_labeled,
                        epoch=epoch
                    )
                    visualized = True
                
    return np.mean(all_mAP)

def main(args):
    # Initialize WandB
    wandb.init(project=args.wandb_project, config=args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Dataset & Loaders
    train_dataset = NucleiDataset(root_dir=args.data_path, transform=transform)
    val_dataset = NucleiDataset(root_dir=args.data_path, transform=val_transform)
    
    # Split into train/val (80/20)
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(0.2 * dataset_size)
    
    train_ds = Subset(train_dataset, indices[split:])
    val_ds = Subset(val_dataset, indices[:split])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # 2. Model, Loss, Optimizer
    model = UNetInstanceSeg(n_channels=3, n_classes=2).to(device)
    criterion = MultiTaskLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 3. Training Loop
    best_mAP = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_mAP = validate(model, val_loader, device, epoch=epoch)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Val mAP: {val_mAP:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Log to WandB
        log_dict = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": train_loss,
            "val_mAP": val_mAP,
            **{f"train_{k}": v for k, v in train_metrics.items()}
        }
        wandb.log(log_dict)
        
        # Save best model
        if val_mAP > best_mAP:
            best_mAP = val_mAP
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mAP': best_mAP,
            }, filename="checkpoints/best_model.pth.tar")
            
        # Regular checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mAP': best_mAP,
            }, filename=f"checkpoints/checkpoint_epoch_{epoch+1}.pth.tar")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Nuclei Instance Segmentation Model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to stage1_train folder")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--wandb_project", type=str, default="cell-segmentation")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    main(args)