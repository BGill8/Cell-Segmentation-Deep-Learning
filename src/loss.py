"""
Custom multi-task loss functions for instance segmentation.
Combines Binary Cross-Entropy (BCE) / Dice Loss for the semantic cell region,
and Mean Squared Error (MSE) or weighted BCE for the distance map/boundary predictions.
This compound loss heavily penalizes the model for merging distinct, touching cells.
"""