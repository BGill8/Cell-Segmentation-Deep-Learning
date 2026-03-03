"""
Inference and post-processing script for the adapted U-Net.
Applies the Marker-Controlled Watershed algorithm to the model's predicted 
distance maps and probability outputs. This algorithmic step converts the 
continuous deep learning outputs into discrete, uniquely labeled cell instances.
"""