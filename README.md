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
├── data-science-bowl-2018/
│   ├── stage1_test/
│   ├── stage1_train/
│   ├── stage2_test_final/
│   ├── stage1_sample_submission.csv
│   ├── stage1_solution.csv
│   ├── stage1_train_labels.csv
│   ├── stage2_sample_submission_final.csv
│   └── ...
├── .gitignore
└── README.md
```

## Install Dependencies
pip install -r requirements.txt

## Data

The data for this project is sourced from the **2018 Data Science Bowl** on Kaggle:
[Data Science Bowl 2018 Data](https://www.kaggle.com/competitions/data-science-bowl-2018/data)
