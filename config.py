"""
config.py — Global Configuration & Hyperparameters
Smart Waste Segregation System

Central configuration file containing all hyperparameters, paths,
and settings used across the entire pipeline.
"""

import os
import torch

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRASHNET_DIR = os.path.join(DATA_DIR, "trashnet")
GARBAGE_CLASSIFICATION_DIR = os.path.join(DATA_DIR, "garbage_classification")
UNIFIED_DATA_DIR = os.path.join(DATA_DIR, "unified")
YOLO_DATA_DIR = os.path.join(DATA_DIR, "yolo")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "models")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")

# ============================================================
# DATASET
# ============================================================
# TrashNet 6 classes (primary label set)
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASSES)

# Kaggle Garbage Classification class mapping to TrashNet classes
# Maps Kaggle class names → TrashNet class names (or None to skip)
GARBAGE_CLASS_MAP = {
    "cardboard": "cardboard",
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "trash": "trash",
    # Additional Kaggle classes mapped to nearest TrashNet class
    "biological": "trash",       # Organic waste → trash
    "battery": "metal",          # Batteries → metal
    "clothes": "trash",          # Textiles → trash
    "shoes": "trash",            # Shoes → trash
    "white-glass": "glass",     
    "brown-glass": "glass",     
    "green-glass": "glass",     
}

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================
# IMAGE PREPROCESSING
# ============================================================
IMG_SIZE = 224                  # Input image size (224x224 for ResNet50)
IMG_CHANNELS = 3                # RGB channels

# ImageNet normalization (used for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================
# DATA AUGMENTATION
# ============================================================
AUGMENTATION = {
    "horizontal_flip_prob": 0.5,
    "rotation_degrees": 30,           # [-30°, +30°]
    "brightness_variation": 0.2,
    "gaussian_blur_kernel": 3,
    "random_crop_size": 224,
    "resize_size": 256,
}

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
# General
BATCH_SIZE = 32
RANDOM_SEED = 42
NUM_WORKERS = 4                 # DataLoader workers

# Adam Optimizer
ADAM_BETAS = (0.9, 0.999)
WEIGHT_DECAY = 1e-4             # L2 regularization (λ = 1e-4)

# Custom CNN
CNN_EPOCHS = 50
CNN_LEARNING_RATE = 1e-3
CNN_EARLY_STOPPING_PATIENCE = 10

# ResNet50 Transfer Learning
RESNET_PHASE1_EPOCHS = 5        # Train head only
RESNET_PHASE1_LR = 1e-3
RESNET_PHASE2_EPOCHS = 25       # Fine-tune top layers
RESNET_PHASE2_LR = 1e-5
RESNET_UNFREEZE_LAYERS = 2      # Number of ResNet blocks to unfreeze

# YOLOv8
YOLO_EPOCHS = 50
YOLO_IMG_SIZE = 640
YOLO_BATCH_SIZE = 16
YOLO_MODEL = "yolov8n.pt"      # Nano model for efficiency

# Learning Rate Scheduler
LR_SCHEDULER = "cosine_annealing"  # Cosine Annealing
LR_T_MAX = 50                      # T_max for CosineAnnealingLR

# Cross-validation
NUM_FOLDS = 5

# ============================================================
# FEATURE EXTRACTION (Hybrid Model)
# ============================================================
FEATURE_DIM = 2048              # ResNet50 GAP output dimension
SELECTED_FEATURES = 128         # After Chi-square selection
FEATURE_SELECTION_METHOD = "chi2"

# ============================================================
# TRADITIONAL ML HYPERPARAMETERS
# ============================================================
# SVM (Cubic Polynomial Kernel)
SVM_PARAMS = {
    "kernel": "poly",
    "degree": 3,                # Cubic
    "C": 1.0,
    "decision_function_shape": "ovo",   # One-vs-One
    "probability": True,
    "random_state": RANDOM_SEED,
}

# KNN
KNN_PARAMS = {
    "n_neighbors": 5,
    "weights": "distance",
    "metric": "minkowski",
    "n_jobs": -1,
}

# Decision Tree
DT_PARAMS = {
    "criterion": "gini",
    "max_depth": 20,
    "min_samples_split": 5,
    "random_state": RANDOM_SEED,
}

# ============================================================
# DEVICE CONFIGURATION
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory
else:
    GPU_NAME = "N/A"
    GPU_MEMORY = 0

# ============================================================
# LOGGING
# ============================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def print_config():
    """Print current configuration summary."""
    print("=" * 60)
    print("SMART WASTE SEGREGATION SYSTEM — CONFIGURATION")
    print("=" * 60)
    print(f"  Device:           {DEVICE}")
    print(f"  GPU:              {GPU_NAME}")
    print(f"  GPU Memory:       {GPU_MEMORY:.1f} GB")
    print(f"  Image Size:       {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch Size:       {BATCH_SIZE}")
    print(f"  Num Classes:      {NUM_CLASSES}")
    print(f"  Classes:          {CLASSES}")
    print(f"  CNN Epochs:       {CNN_EPOCHS}")
    print(f"  ResNet Epochs:    {RESNET_PHASE1_EPOCHS}+{RESNET_PHASE2_EPOCHS}")
    print(f"  YOLO Epochs:      {YOLO_EPOCHS}")
    print(f"  K-Folds:          {NUM_FOLDS}")
    print(f"  Random Seed:      {RANDOM_SEED}")
    print(f"  L2 Lambda:        {WEIGHT_DECAY}")
    print(f"  LR Scheduler:     {LR_SCHEDULER}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
