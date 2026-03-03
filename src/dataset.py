"""
PyTorch Dataset class for the Data Science Bowl 2018 instance segmentation dataset.
Instead of a single merged binary mask, this loader processes individual nucleus 
masks to generate distance transforms (distance maps) and boundary masks. 
These targets force the model to learn the spatial separation between touching cells.
"""