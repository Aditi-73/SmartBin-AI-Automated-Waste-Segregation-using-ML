# Smart Waste Segregation System

A complete implementation of smart waste segregation using hybrid AI models.
Based on the methodology from "Smart Waste Segregation with Machine Learning: A Pathway to Sustainable and Intelligent Cities".

## Features

1. **Convolutional Neural Network (Custom CNN)** - Baseline classification model.
2. **Transfer Learning with ResNet50** - High accuracy feature extraction.
3. **Traditional Machine Learning Models** - SVM, KNN, Decision Tree.
4. **Hybrid Model (ResNet50 + SVM)** - State-of-the-art final model with Chi-Square feature selection.
5. **YOLOv8** - Object detection for real-time waste detection.

## Prerequisites

- Python 3.10
- CUDA capable GPU (highly recommended for performance)

## Installation

```bash
# Optional: Create a virtual environment using Python 3.10
py -3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Configuration

The system automatically downloads the datasets using Kaggle API. Ensure you have your `~/.kaggle/kaggle.json` configured. Alternatively, the script will guide you to download them manually if it fails.

## Usage

You can run the full pipeline or specific components using the `--mode` argument in `main.py`.

```bash
# 1. Prepare Data Only
python main.py --mode data

# 2. Train Classification Models (CNN, ResNet50, Hybrid, Traditional ML)
python main.py --mode train

# 3. Train YOLOv8 Model
python main.py --mode yolo

# 4. Run Everything
python main.py --mode full
```

## Outputs
All generated artifacts are saved in the `outputs/` directory:
- `outputs/models/`: Saved model weights (`.pth` and `.joblib`).
- `outputs/figures/`: Performance graphs, confusion matrices, ROC curves.
- `outputs/results/`: Final comparison reports (`.csv`, `.md`).
- `outputs/logs/`: Detailed execution logs.
