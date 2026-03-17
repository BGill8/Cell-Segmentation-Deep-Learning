import numpy as np
import pandas as pd
import os
from skimage import measure

def rle_encoding(x):
    """Converts a binary mask to Kaggle's Run-Length Encoding format."""
    # Transpose first to read column by column (top-to-bottom, left-to-right)
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): 
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join(str(val) for val in run_lengths)

def main():
    pred_dir = 'outputs/predictions' # Update if your path is different
    submission_data = []

    print("Generating Run-Length Encodings... this might take a minute.")
    
    for file in os.listdir(pred_dir):
        if file.endswith('_mask.npy'):
            image_id = file.replace('_mask.npy', '')
            mask = np.load(os.path.join(pred_dir, file))
            
            # Separate the single image mask into individual cell instances
            # (Assuming your mask is binary > 0)
            labeled_mask = measure.label(mask > 0) 
            
            # Loop through every single cell found in the image
            for i in range(1, labeled_mask.max() + 1):
                instance_mask = (labeled_mask == i).astype(np.uint8)
                rle = rle_encoding(instance_mask)
                
                if rle: # Only add if the mask isn't empty
                    submission_data.append([image_id, rle])

    # Save to Kaggle format
    df = pd.DataFrame(submission_data, columns=['ImageId', 'EncodedPixels'])
    df.to_csv('outputs/submission.csv', index=False)
    print(f"Success! Saved {len(df)} individual nuclei to outputs/submission.csv")

if __name__ == '__main__':
    main()