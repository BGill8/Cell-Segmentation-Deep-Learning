"""
Evaluation metrics for instance segmentation performance.
Calculates Mean Average Precision (mAP) across various Intersection over Union (IoU) 
thresholds, rather than standard semantic IoU, to accurately score both 
the detection (counting) and the exact boundary pixel accuracy of individual cells.
"""

import numpy as np
def compute_iou(mask1, mask2):
    """Computes Intersection over Union (IoU) between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union
def calculate_precision(pred_labeled, true_labeled, threshold):
    """
    Calculates precision at a specific IoU threshold.
    
    Args:
        pred_labeled: [H, W] numpy array where each instance has a unique ID
        true_labeled: [H, W] numpy array where each instance has a unique ID
        threshold: IoU threshold (e.g., 0.5)
        
    Returns:
        precision: TP / (TP + FP + FN)
    """
    # Get unique IDs (excluding background 0)
    pred_ids = np.unique(pred_labeled)[1:]
    true_ids = np.unique(true_labeled)[1:]
    if len(true_ids) == 0:
        return 1.0 if len(pred_ids) == 0 else 0.0
    if len(pred_ids) == 0:
        return 0.0
    # Compute IoU matrix between all pairs
    # Rows: True instances, Cols: Predicted instances
    iou_matrix = np.zeros((len(true_ids), len(pred_ids)))
    
    for i, t_id in enumerate(true_ids):
        true_mask = (true_labeled == t_id)
        for j, p_id in enumerate(pred_ids):
            pred_mask = (pred_labeled == p_id)
            iou_matrix[i, j] = compute_iou(true_mask, pred_mask)
    # Count True Positives (TP)
    # A match is a TP if IoU > threshold
    matches = (iou_matrix > threshold)
    
    tp = 0
    # Greedy matching: each true/pred instance can only be used once
    used_true = np.zeros(len(true_ids), dtype=bool)
    used_pred = np.zeros(len(pred_ids), dtype=bool)
    
    # Sort by IoU to match the best pairs first
    indices = np.argsort(iou_matrix.ravel())[::-1]
    for idx in indices:
        i, j = divmod(idx, len(pred_ids))
        if iou_matrix[i, j] > threshold and not used_true[i] and not used_pred[j]:
            tp += 1
            used_true[i] = True
            used_pred[j] = True
    fp = len(pred_ids) - tp
    fn = len(true_ids) - tp
    return tp / (tp + fp + fn)
def mean_average_precision(pred_labeled, true_labeled, thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    Calculates the mean precision across multiple IoU thresholds.
    
    Args:
        pred_labeled: [H, W] labeled instance map
        true_labeled: [H, W] labeled instance map
        thresholds: List of IoU thresholds to average over
        
    Returns:
        mAP: The mean average precision
    """
    precisions = []
    for t in thresholds:
        precisions.append(calculate_precision(pred_labeled, true_labeled, t))
    
    return np.mean(precisions)

