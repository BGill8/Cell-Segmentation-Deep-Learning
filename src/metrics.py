"""
Evaluation metrics for instance segmentation performance.
Calculates Mean Average Precision (mAP) across various Intersection over Union (IoU) 
thresholds, rather than standard semantic IoU, to accurately score both 
the detection (counting) and the exact boundary pixel accuracy of individual cells.
"""