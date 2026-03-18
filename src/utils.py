import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Saves the model and optimizer state to a file."""
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer=None, lr=None):
    """Loads a checkpoint and updates the model and optimizer states."""
    print(f"=> Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    # If a new learning rate is provided, override the one in the optimizer
    if lr is not None and optimizer is not None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
    return checkpoint.get("epoch", 0), checkpoint.get("best_mAP", 0.0)

def visualize_prediction(image, true_mask, pred_semantic, pred_dist, pred_instances, epoch=None, save_path=None):
    """
    Creates a 5-panel visualization of the model's performance.
    """
    # Convert image tensor to numpy [H, W, 3] if needed
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Panels 1-4
    axes[0].imshow(image); axes[0].set_title("Original Image"); axes[0].axis("off")
    axes[1].imshow(true_mask, cmap="gray"); axes[1].set_title("Ground Truth"); axes[1].axis("off")
    axes[2].imshow(pred_semantic, cmap="jet"); axes[2].set_title("Pred: Semantic"); axes[2].axis("off")
    axes[3].imshow(pred_dist, cmap="hot"); axes[3].set_title("Pred: Distance"); axes[3].axis("off")
    
    # 5. Final Watershed Instances with High Contrast
    num_labels = int(pred_instances.max())
    if num_labels > 0:
        # Create a shuffled color map for maximum contrast between neighbors
        rng = np.random.default_rng() # Random seed for each call
        colors = rng.random((num_labels + 1, 3))
        colors[0] = [0, 0, 0] # Background is black
        
        # Apply the color map
        colored_instances = colors[pred_instances.astype(int)]
        axes[4].imshow(colored_instances)
    else:
        axes[4].imshow(np.zeros_like(pred_instances), cmap="gray")
        
    axes[4].set_title(f"Instances ({num_labels})")
    axes[4].axis("off")
    
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    if wandb.run is not None: wandb.log({"visual_results": wandb.Image(plt)})
    plt.close(fig)

def log_metrics_to_wandb(metrics, epoch):
    """Helper to log a dictionary of metrics to WandB."""
    if wandb.run is not None:
        wandb.log(metrics, step=epoch)