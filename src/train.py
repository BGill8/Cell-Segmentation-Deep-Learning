"""
Main training script.
Initializes the model, dataloaders, loss functions, and optimizer.
Contains the training loop and logs metrics directly to Weights & Biases (WandB).
"""
#Check if an NVIDIA GPU is available, otherwise fallback to CPU

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")
from torch.utils.data import DataLoader
import torch

from dataset import NucleiDataset, transform
from model import UNetInstanceSeg
from loss import MultiTaskLoss

from utils import save_checkpoint, visualize_prediction, log_metrics_to_wandb
from evaluate import run_inference
import wandb

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(
        project="cell-segmentation",
        name="unet-instance-seg",

        config={"batch_size": 16, "learning_rate": 1e-4, "epochs": 30},

    )

    train_dataset = NucleiDataset(
        root_dir="data/data-science-bowl-2018/stage1_train",
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,       # start with 1 because instance masks vary
        shuffle=True,
        num_workers=2
    )


    #before testing to check:
    for images, masks in train_loader:
        print(images.shape)
        print(masks.shape)
        break

    model = UNetInstanceSeg(n_channels=3, n_classes=2).to(device)
    wandb.watch(model, log="all", log_freq=10)
    criterion = MultiTaskLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    num_epochs = 30
    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for images, targets in train_loader:    

            #this is where you define the training...
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            predictions = model(images)
            loss, loss_dict = criterion(predictions, targets)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1} / {num_epochs}] - Loss: {avg_loss:.4f}")

        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({"epoch": epoch + 1, "train_loss": avg_loss, "lr": current_lr})
        scheduler.step()


        log_metrics_to_wandb({"train loss": avg_loss}, epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(                                    #IF LOSS IS THE BEST SO FAR, SAVE BEST CHECKPOINT
                {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_loss": best_loss,
                },
                filename="best_checkpoint.pth.tar",
            )


        save_checkpoint(
            {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_mAP": 0.0,
            },
            filename="checkpoint.pth.tar",
        )



    

    print("Checkpoint saved. Running inference on a sample image...")

    model.eval()
    sample_image, sample_target = train_dataset[0]

    with torch.no_grad():
        output = model(sample_image.unsqueeze(0).to(device))
        pred_semantic = torch.sigmoid(output[0, 0]).cpu().numpy()
        pred_dist = output[0, 1].cpu().numpy()

    pred_instances = run_inference(model, sample_image)
    true_mask = sample_target[0].cpu().numpy()

    print("Saving visualization to prediction_example.png...")

    visualize_prediction(
        image=sample_image,
        true_mask=true_mask,
        pred_semantic=pred_semantic,
        pred_dist=pred_dist,
        pred_instances=pred_instances,
        save_path="prediction_example.png",
    )

    print("Visualization saved.")
    wandb.log({"prediction_example": wandb.Image("prediction_example.png")})
    wandb.finish()


if __name__ == "__main__":
    main()
    #FINAL GOAL:
    #For each image, model must predict
    #Instance masks

    #EX: if image has 10 nuclei, must output 18 masks




    #NOTES for implementing the UNET structure:

    #They did not feed (N, H, W) instance masks directly into U-Net, because the number of nuclei N changes per image.
    #Instead, they converted the instance masks into fixed-size pixel maps so the network always predicts the same output shape.
