"""
Inference and post-processing script for the adapted U-Net.
Applies the Marker-Controlled Watershed algorithm to the model's predicted 
distance maps and probability outputs. This algorithmic step converts the 
continuous deep learning outputs into discrete, uniquely labeled cell instances.
"""
# Check if an NVIDIA GPU is available, otherwise fallback to CPU
import torch
import numpy as np
import skimage
from skimage import measure
from skimage.segmentation import watershed
from skimage.measure import label
from scipy import ndimage as maxi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def post_process_watershed(semantic_logits, distance_map, 
                           semantic_threshold=0.5, 
                           dist_threshold=0.3):
    """
    Applies Marker-Controlled Watershed to separate cell instances.
    
    Args:
        semantic_logits: [H, W] tensor of raw logits from Channel 0
        distance_map: [H, W] tensor of predicted distances from Channel 1
        semantic_threshold: Confidence to consider a pixel as 'cell'
        dist_threshold: Threshold on distance map to find cell 'cores' (markers)
        
    Returns:
        instance_mask: [H, W] numpy array where each cell has a unique integer ID
    """
    # 1. Convert to numpy and apply sigmoid to get probabilities
    prob_mask = torch.sigmoid(semantic_logits).cpu().numpy()
    dist_map = distance_map.cpu().numpy()

    # 2. Threshold the semantic mask to get the 'territory'
    binary_mask = (prob_mask > semantic_threshold).astype(np.uint8)

    # 3. Find markers (the 'seeds' for each cell)
    # We use the distance map to find the centers of cells.
    # Pixels above dist_threshold are considered part of a cell's core.
    markers_binary = (dist_map > dist_threshold).astype(np.uint8)
    
    # 4. Label the markers (each discrete seed gets a unique ID)
    markers = label(markers_binary)

    # 5. Apply Watershed
    # We use -dist_map because watershed fills 'basins'. 
    # Negating the distance map turns cell centers into the deepest points.
    labels = watershed(-dist_map, markers, mask=binary_mask)

    return labels

def run_inference(model, image_tensor):
    """
    Runs a single image through the model and post-processes it.
    
    Args:
        model: The trained UNetInstanceSeg model
        image_tensor: [3, H, W] input image
        
    Returns:
        instances: [H, W] uniquely labeled cell instances
    """
    model.eval()
    with torch.no_grad():
        # Add batch dimension: [1, 3, H, W]
        input_batch = image_tensor.unsqueeze(0).to(device)
        
        # Forward pass: [1, 2, H, W]
        output = model(input_batch)
        
        # Extract channels [H, W]
        semantic_logits = output[0, 0, :, :]
        distance_map = output[0, 1, :, :]
        
        # Post-process
        instances = post_process_watershed(semantic_logits, distance_map)
        
    return instances

if __name__ == "__main__":
    print(f"Post-processing module initialized on {device}")