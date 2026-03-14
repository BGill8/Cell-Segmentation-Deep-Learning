"""
PyTorch Dataset class for the Data Science Bowl 2018 instance segmentation dataset.
Instead of a single merged binary mask, this loader processes individual nucleus
masks to generate distance transforms (distance maps) and boundary masks.
These targets force the model to learn the spatial separation between touching cells.
"""

import torch
import torchvision
import torchvision.transforms as transforms #basic aug tools
import os
import cv2
import numpy as np

import albumentations as A  #advanced image aug library (to apply same transform to image + mult mask)
from albumentations.pytorch import ToTensorV2 #convert np array to PyTorch tensors

#WE ARE DOING INSTANCE SEGMENTATON
  #Load image
  #Load multiple mask PNGs
  #Stack them
  #Apply SAME random transform
  #Return them

#if training:
#   apply spatial transforms to (image, binary_mask)
#    apply intensity transforms to image only

#Return
  #image: (C, H, W)
  #masks: (N, H, W)

#creates trasnsform pipeline
#each time called, randomly decides
transform = A.Compose([
    # ----- Geometric / spatial (applies to image + masks) -----
    A.Resize(256, 256),

    #50% chance
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),

    #40% chance of rotating
    A.Affine(
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        scale=(0.90, 1.10),
        rotate=(-15, 15),
        p=0.4,
    ),

    #15% chance block triggers, and picks one of these
    #A.OneOf([
        #A.ElasticTransform(alpha=60, sigma=8),
        #A.Perspective(scale=(0.02, 0.05)),
        #A.PiecewiseAffine(scale=(0.01, 0.03)),
        #A.OpticalDistortion(distort_limit=0.05),
    #], p=0.15),

    # ----- Intensity / appearance (image only) -----

    #20% chance block triggers, and picks one of these
    A.OneOf([
        A.GaussNoise(std_range=(0.04, 0.10)),
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
    ], p=0.2),  

    A.OneOf([
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.0)),
        A.Emboss(alpha=(0.1, 0.3), strength=(0.2, 0.5)),
    ], p=0.1),  

    

    #10% chance of shuffling
    A.ChannelShuffle(p=0.1),
    #5 chance of grayscaling
    A.ToGray(p=0.05),
])

#create custom dataset
class NucleiDataset(torch.utils.data.Dataset):

  #runs once you create dataset
  def __init__(self, root_dir, transform=None):
    self.root_dir = root_dir  #stores dataset path
    self.ids = os.listdir(root_dir) #lists all folders inside root_dir (each folder = one image)
    self.transform = transform  #stores aug pipeline


  #stores number of training samples
  def __len__(self):
    return len(self.ids)

  #runs everytime DataLoader asks for one sample
  #created so that the output images and mask are not saved on the disk but instead created in memory 
  #and returned to the Dataloader each batch
  def __getitem__(self, idx):

    image_id = self.ids[idx]  #gets folder name for sample

    #Load image
    img_path = os.path.join(self.root_dir, image_id, "images", image_id + ".png") #builds image path (ex: data/data-science-bowl-2018/stage1_train/abc123/images/abc123.png)
    image = cv2.imread(img_path)  #loads image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #converts it to RGB

    #Load all masks
    mask_dir = os.path.join(self.root_dir, image_id, "masks") #finds all individual nucleus mask PNGs (data/data-science-bowl-2018/stage1_train/abc123/masks/)
    mask_files = os.listdir(mask_dir)

    masks = []
    for m in mask_files:  #loop through each mask
      mask_path = os.path.join(mask_dir, m)
      mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  #loads mask as a single channel image
      mask = (mask > 0).astype(np.uint8)  #convert to binary
      masks.append(mask)  #adds the mask to masks list

    #USED TO STACK THE MASKS
    masks = np.stack(masks, axis=0)  # (N, H, W)

    #Apply augmentation to image and its masks
    if self.transform:
       #uses transform function to randomly augment images AND along with its masks
       #auto knows that images + mask for spatial changes and only images for intensity
      augmented = self.transform(image=image, masks=list(masks))  
      image = augmented["image"]
      masks = np.stack(augmented["masks"], axis=0)

    #collapse all nucleus masks into one image
    semantic_mask = (np.sum(masks, axis=0) > 0).astype(np.float32)

    #creates an empty (H, W) map full of zeroes
    distance_map = np.zeros_like(semantic_mask, dtype=np.float32)

    #loops through each individual nucelus masks and compute a distance transformation
    for mask in masks: 
      mask_uint8 = mask.astype(np.uint8)
      dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)

      #normalize distance target
      max_dist = dist.max()
      if max_dist > 0:
        dist = dist / max_dist
      distance_map = np.maximum(distance_map, dist)

  
    # Convert to tensors
    image = torch.tensor(image).permute(2, 0, 1).float() / 255.0   #permute so shape is (C, H, W)
    
    #converts to tensors
    target = np.stack([semantic_mask, distance_map], axis=0)
    target = torch.tensor(target).float()

    return image, target

    #model receives:
    #image (3, H, W) --> (number of channels, height of image, width of image)

    #target (2, H, W) --> means the target is no longer “one separate mask per nucleus.” Instead, it is two image-sized maps stacked together:
      #Channel 0: one combined semantic mask for all nuclei
      #Channel 1: one distance map
