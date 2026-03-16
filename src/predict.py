"""
Prediction script for Cell Instance Segmentation.
This script loads a trained model and runs inference on images from either 
stage1_train or stage2_test_final. It can save individual instance masks, 
visualizations, or a Kaggle-style submission CSV.
"""

import os
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm

# Force matplotlib to use a non-interactive backend for HPC stability
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local imports
from model import UNetInstanceSeg
from evaluate import post_process_watershed

def load_model(checkpoint_path, device):
    """Loads a trained UNetInstanceSeg model from a checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    model = UNetInstanceSeg(n_channels=3, n_classes=2).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both full state dict and partial state dict
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_image(img_path, target_size=(256, 256)):
    """Loads and preprocesses an image for the model."""
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Could not read image at: {img_path}")
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Save original size for post-processing if needed
    original_size = image.shape[:2]
    
    # Resize to model input size
    image_resized = cv2.resize(image, (target_size[1], target_size[0]))
    
    # Normalize and convert to tensor [1, 3, H, W]
    image_tensor = torch.tensor(image_resized).permute(2, 0, 1).float() / 255.0
    return image_tensor, image_resized, original_size

def run_prediction(model, image_tensor, device):
    """Runs a single prediction and returns raw outputs."""
    with torch.no_grad():
        input_batch = image_tensor.unsqueeze(0).to(device)
        output = model(input_batch)
        
        semantic_logits = output[0, 0, :, :]
        distance_map = output[0, 1, :, :]
        
    return semantic_logits, distance_map

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Model
    model = load_model(args.checkpoint, device)
    
    # 2. Get list of image folders
    image_ids = [i for i in os.listdir(args.input_dir) if not i.startswith('.')]
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Found {len(image_ids)} images in {args.input_dir}")
    
    for img_id in tqdm(image_ids):
        # Path to the actual image file inside its ID folder
        img_path = os.path.join(args.input_dir, img_id, "images", img_id + ".png")
        if not os.path.exists(img_path):
            continue
            
        try:
            # Preprocess
            image_tensor, image_rgb, original_size = preprocess_image(img_path)
            
            # Inference
            semantic_logits, distance_map = run_prediction(model, image_tensor, device)
            
            # Post-process (Watershed)
            instances = post_process_watershed(
                semantic_logits, 
                distance_map,
                semantic_threshold=args.semantic_threshold,
                dist_threshold=args.dist_threshold
            )
            
            # Resize back to original if requested
            if args.save_original_size:
                instances = cv2.resize(
                    instances.astype(np.float32), 
                    (original_size[1], original_size[0]), 
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int32)
                
            # Save results
            if args.visualize:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                # Use original image if possible for high-res visualization
                if args.save_original_size:
                    axes[0].imshow(image_rgb if original_size[0] == 256 else cv2.imread(img_path)[:,:,::-1])
                else:
                    axes[0].imshow(image_rgb)
                
                axes[0].set_title("Original Image")
                axes[0].axis("off")
                
                axes[1].imshow(instances, cmap="nipy_spectral")
                axes[1].set_title(f"Predicted Instances ({instances.max()} nuclei)")
                axes[1].axis("off")
                
                plt.savefig(os.path.join(args.output_dir, f"{img_id}_pred.png"))
                plt.close(fig) # Explicitly pass fig to close
                
            # Optional: Save raw instance mask as a numpy file
            if args.save_masks:
                np.save(os.path.join(args.output_dir, f"{img_id}_mask.npy"), instances)
        
        except Exception as e:
            print(f"Error processing {img_id}: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Nuclei Instances")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to data folder (e.g. stage1_train)")
    parser.add_argument("--output_dir", type=str, default="outputs/predictions")
    parser.add_argument("--semantic_threshold", type=float, default=0.5)
    parser.add_argument("--dist_threshold", type=float, default=0.3)
    parser.add_argument("--visualize", action="store_true", help="Save visualization plots")
    parser.add_argument("--save_masks", action="store_true", help="Save raw instance masks as .npy files")
    parser.add_argument("--save_original_size", action="store_true", help="Resize prediction back to original image size")
    
    args = parser.parse_args()
    main(args)
