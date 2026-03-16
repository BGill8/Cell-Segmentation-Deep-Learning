#This program runs ensemble methods on 3 checkpoints

#how it works high level is when you run train.py 3 times to which it saves
#the outputs in 3 models

#running this program will average their outputs and run watershed on the averaged output 

#running watershed will turn the maps into individual cell instances

import torch

from dataset import NucleiDataset, val_transform
from evaluate import run_ensemble_inference
from metrics import mean_average_precision
from model import UNetInstanceSeg


def load_models(checkpoint_paths, device):
    models = []
    for path in checkpoint_paths:
        model = UNetInstanceSeg(n_channels=3, n_classes=2).to(device)
        # Set weights_only=False to allow loading custom metadata/numpy scalars
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    return models


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_paths = [
        "best_checkpoint_1.pth.tar",
        "best_checkpoint_2.pth.tar",
        "best_checkpoint_3.pth.tar",
    ]

    models = load_models(checkpoint_paths, device)

    val_dataset = NucleiDataset(
        root_dir="data/data-science-bowl-2018/stage1_train",
        transform=val_transform,
        return_instance_map=True,
    )

    image, target, true_instances = val_dataset[0]
    pred_instances = run_ensemble_inference(models, image)

    map_score = mean_average_precision(pred_instances, true_instances.numpy())

    print("Predicted instances shape:", pred_instances.shape)
    print("Number of predicted nuclei:", pred_instances.max())
    print(f"Ensemble mAP: {map_score:.4f}")


if __name__ == "__main__":
    main()
