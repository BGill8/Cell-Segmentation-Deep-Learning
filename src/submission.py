import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm

def rle_encoding(x):
    """
    Converts a binary mask to Kaggle's Run-Length Encoding format.
    x: binary mask (numpy array)
    """
    # Transpose first to read column by column (Kaggle requirement)
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): 
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join(str(val) for val in run_lengths)

def main(args):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    submission_data = []

    # Loop through every directory passed into the script.
    for current_dir in args.pred_dirs:
        # Get all mask files in the current prediction directory
        if not os.path.exists(current_dir):
            print(f"Warning: Directory {current_dir} not found. Skipping.")
            continue
            
        mask_files = [f for f in os.listdir(current_dir) if f.endswith('_mask.npy')]
        print(f"Generating RLE for {len(mask_files)} images from {current_dir}...")

        for file in tqdm(mask_files):
            image_id = file.replace('_mask.npy', '')

            # Load the ALREADY LABELED instance mask from Watershed
            # Each nucleus has a unique ID (1, 2, 3...)
            labeled_mask = np.load(os.path.join(current_dir, file))

            # Number of unique nuclei found in this image
            num_nuclei = labeled_mask.max()

            # >>> FIXED: Kaggle requires an empty row if no nuclei are found! <<<
            if num_nuclei == 0:
                submission_data.append([image_id, ''])
                continue

            # Loop through every single nucleus found in the image
            for i in range(1, num_nuclei + 1):
                instance_mask = (labeled_mask == i).astype(np.uint8)
                rle = rle_encoding(instance_mask)

                if rle: # Only add if the mask isn't empty
                    submission_data.append([image_id, rle])

    # Create DataFrame and save to CSV
    df = pd.DataFrame(submission_data, columns=['ImageId', 'EncodedPixels'])
    df.to_csv(args.output_file, index=False)

    print(f"\nSuccess! Saved {len(df)} total nuclei/rows to {args.output_file}")
    print("This file is now ready for submission to Kaggle.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Kaggle Submission CSV")
    
    # Updated to accept multiple directories
    parser.add_argument("--pred_dirs", nargs='+', 
                        default=["outputs/stage1_predictions", "outputs/stage2_predictions"], 
                        help="List of directories containing the _mask.npy files")
    
    # >>> CHANGED: Unmistakable new file name <<<
    parser.add_argument("--output_file", type=str, default="outputs/FINAL_Kaggle_Submission.csv",
                        help="Path to the final CSV file")

    args = parser.parse_args()
    main(args)