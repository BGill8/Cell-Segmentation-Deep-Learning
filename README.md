# Cell-Segmentation-Deep-Learning
Semantic Nuclei Cell Segmentation Project for AI535 Final Project


**Team Members:** 
* Brandon Gill
* Andy Bui

## Description

Cell segmentation is a critical process for defining boundaries in microscopic images, enabling quantitative analysis of cell counts, shapes, and molecular content. By accurately identifying individual cells, this technology supports drug discovery, disease research, and spatial tissue analysis, ultimately helping to improve cancer diagnosis and treatment strategies.

* **Architecture:** U-Net
* **Loss Function:** Binary Cross-Entropy (BCE) + Dice Loss
* **Evaluation Metric:** Intersection over Union (IoU)
* **Experiment Tracking:** Weights & Biases (WandB)

## Project Structure

```text
Cell-Segmentation-Deep-Learning/
├── app/
│   └── app.py
├── checkpoints/
│   └── unet_best_model.pth
├── data/
│   ├── augmented/
│   └── data-science-bowl-2018/
│       ├── stage1_test/
│       ├── stage1_train/
│       ├── stage2_test_final/
│       ├── stage1_sample_submission.csv
│       ├── stage1_solution.csv
│       ├── stage1_train_labels.csv
│       └── stage2_sample_submission_final.csv
├── notebooks/
│   └── 01_data_exploration.ipynb
├── outputs/
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── loss.py
│   ├── metrics.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── .gitignore
├── README.md
├── requirements.txt
└── train.slurm
```

## Install Dependencies
pip install -r requirements.txt

## Data

The data for this project is sourced from the **2018 Data Science Bowl** on Kaggle:
[Data Science Bowl 2018 Data](https://www.kaggle.com/competitions/data-science-bowl-2018/data)


## OSU HPC Cheat Sheet
# Running on the OSU HPC Cluster

This project is configured to run on the Oregon State University high-performance computing (HPC) cluster using the SLURM workload manager. Since we are performing **instance segmentation**, utilizing the cluster's GPU nodes is highly recommended for training.

> **Note:** Instance segmentation requires significantly more computational overhead than semantic segmentation — especially when computing watershed algorithms or running models like Mask R-CNN or StarDist. The HPC cluster's GPU nodes are the right tool for this workload.

---

## 1. Connecting & Setup

First, SSH into the cluster (ensure you are on the OSU VPN if off-campus):

```bash
ssh your_onid@submit.hpc.engr.oregonstate.edu
```

Clone the repository into your home directory or scratch space:

```bash
git clone https://github.com/BGill8/Cell-Segmentation-Deep-Learning.git
cd Cell-Segmentation-Deep-Learning
```

---

## 2. Environment Configuration

Do **not** install packages directly to your base environment. Load the necessary CUDA and Python modules, then create a virtual environment.

```bash
# Load necessary modules (adjust versions based on current HPC availability)
module load python/3.10
module load cuda/11.8

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Data Management

> **Important:** Do not train directly out of your home directory. Move the 2018 Data Science Bowl dataset to the high-speed scratch storage to prevent I/O bottlenecks during the data loader's mask-merging process.

```bash
# Move data to scratch space
cp -r data/data-science-bowl-2018 /scratch/your_onid/data-science-bowl-2018
```

> **Note:** Ensure you update your dataset paths in `src/dataset.py` or pass the new scratch path as an argument when running your training script.

---

## 4. Weights & Biases (WandB) Login

Before submitting a job, you must authenticate WandB on the cluster to track instance segmentation metrics (training loss, IoU, mask visualizations, etc.):

```bash
wandb login
```

Paste your API key when prompted.

---

## 5. Submitting a Training Job

> **Never** run `python src/train.py` directly on the login node. Always submit a batch job using SLURM.

Create a file named `train.slurm` in the project root:

```bash
#!/bin/bash
#SBATCH --job-name=nuclei_instance_seg
#SBATCH --partition=gpu            # Specify the GPU partition
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --cpus-per-task=4          # Number of CPU cores for data loading
#SBATCH --mem=32G                  # Memory required
#SBATCH --time=12:00:00            # Maximum time limit (hrs:min:sec)
#SBATCH --output=outputs/slurm-%j.out

# Load modules and activate environment
module load python/3.10
module load cuda/11.8
source .venv/bin/activate

# Run the training script
python src/train.py --data_path /scratch/your_onid/data-science-bowl-2018/stage1_train
```

Submit the job to the queue:

```bash
sbatch train.slurm
```

---

## 6. Monitoring Progress

| Method | Command / Location |
|---|---|
| **SLURM queue** | `squeue -u your_onid` |
| **Live logs** | `tail -f outputs/slurm-<JOB_ID>.out` |
| **Metrics dashboard** | WandB project dashboard (loss, IoU, mask visualizations) |

---

## 7. Recommended `argparse` Setup for `src/train.py`

To cleanly pass the scratch directory path (and other hyperparameters) from your SLURM script via the command line, add the following argument parser to `src/train.py`:

```python
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train nuclei instance segmentation model")

    # Data
    parser.add_argument("--data_path", type=str,
                        default="data/data-science-bowl-2018/stage1_train",
                        help="Path to training data (use scratch path on HPC)")

    # Training hyperparameters
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=4,
                        help="DataLoader workers — match --cpus-per-task in SLURM script")

    # Experiment tracking
    parser.add_argument("--wandb_project", type=str, default="cell-segmentation")
    parser.add_argument("--run_name",      type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # e.g., dataset = CellDataset(root=args.data_path)
```

Then in your SLURM script, you can override any default at submission time:

```bash
python src/train.py \
    --data_path /scratch/your_onid/data-science-bowl-2018/stage1_train \
    --epochs 100 \
    --batch_size 16 \
    --run_name "maskrcnn_run1"
```

---

## Quick Reference Cheatsheet

```bash
# Check available modules
module spider python
module spider cuda

# Check job queue
squeue -u your_onid

# Cancel a job
scancel <JOB_ID>

# Check cluster node availability
sinfo -p gpu

# Check your scratch storage usage
du -sh /scratch/your_onid/
```