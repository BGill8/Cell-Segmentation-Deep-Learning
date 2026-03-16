import os
import random

import cv2
import numpy as np
import torch

from model import UNetInstanceSeg
from evaluate import run_inference
from utils import visualize_prediction, load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_test_image(image_path, size=(256, 256)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)

    image_tensor = torch.tensor(image).permute(2, 0, 1).float() / 255.0
    return image_tensor


def pick_random_test_image():
    candidate_dirs = [
        "data/data-science-bowl-2018/stage1_test",
        "data/data-science-bowl-2018/stage2_test_final",
    ]

    test_root = next((path for path in candidate_dirs if os.path.isdir(path)), None)
    if test_root is None:
        raise FileNotFoundError("Could not find stage1_test or stage2_test_final.")

    image_ids = sorted(os.listdir(test_root))
    image_id = random.choice(image_ids)
    image_path = os.path.join(test_root, image_id, "images", f"{image_id}.png")
    return image_id, image_path


def main():
    checkpoint_path = "checkpoints/best_model_mAP.pth.tar"
    image_id, image_path = pick_random_test_image()

    model = UNetInstanceSeg(n_channels=3, n_classes=2).to(device)
    load_checkpoint(checkpoint_path, model)

    image_tensor = load_test_image(image_path)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device))
        pred_semantic = torch.sigmoid(output[0, 0]).cpu().numpy()
        pred_dist = output[0, 1].cpu().numpy()

    pred_instances = run_inference(model, image_tensor)

    # If no ground truth exists, use a blank mask just for visualization
    true_mask = np.zeros_like(pred_semantic)
    save_path = f"test_prediction_{image_id}.png"

    visualize_prediction(
        image=image_tensor,
        true_mask=true_mask,
        pred_semantic=pred_semantic,
        pred_dist=pred_dist,
        pred_instances=pred_instances,
        save_path=save_path,
    )

    print(f"Ran inference on test image: {image_id}")
    print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    main()
