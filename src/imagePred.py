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


def main():
    checkpoint_path = "src/checkpoint.pth.tar"
    image_path = "path/to/test_image.png"

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

    visualize_prediction(
        image=image_tensor,
        true_mask=true_mask,
        pred_semantic=pred_semantic,
        pred_dist=pred_dist,
        pred_instances=pred_instances,
        save_path="test_prediction.png",
    )

    print("Saved visualization to test_prediction.png")


if __name__ == "__main__":
    main()
