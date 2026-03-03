"""
Adapted U-Net architecture for instance segmentation.
Maintains the standard Encoder-Decoder structure with skip connections, 
but the final output layer is modified to predict multiple channels:
typically a core nucleus probability map and a distance map (or boundary map) 
to aid in separating overlapping instances.
"""