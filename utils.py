"""
utils.py — Utility Functions
Smart Waste Segregation System

Helper functions for reproducibility, logging, model saving/loading,
directory management, and common plotting utilities.
"""

import os
import random
import logging
import time
import json
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

import config


def set_seed(seed: int = config.RANDOM_SEED):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (default from config)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set to {seed} for reproducibility.")


def get_device() -> torch.device:
    """
    Get the best available compute device (CUDA GPU or CPU).
    
    Returns:
        torch.device: CUDA device if available, else CPU
    """
    device = config.DEVICE
    if device.type == "cuda":
        logging.info(f"Using GPU: {config.GPU_NAME} ({config.GPU_MEMORY:.1f} GB)")
    else:
        logging.warning("No CUDA GPU detected. Training will be slow on CPU.")
    return device


def setup_logging(log_file: str = None):
    """
    Configure logging to both console and file.
    
    Args:
        log_file: Optional log file path. If None, uses default in LOG_DIR.
    """
    if log_file is None:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(config.LOG_DIR, f"training_{timestamp}.log")

    # Create formatters and handlers
    formatter = logging.Formatter(config.LOG_FORMAT)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info(f"Logging initialized. Log file: {log_file}")


def create_output_dirs():
    """Create all necessary output directories."""
    dirs = [
        config.DATA_DIR,
        config.UNIFIED_DATA_DIR,
        config.YOLO_DATA_DIR,
        config.OUTPUT_DIR,
        config.MODEL_SAVE_DIR,
        config.FIGURE_DIR,
        config.RESULTS_DIR,
        config.LOG_DIR,
        config.FEATURES_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logging.info("Output directories created.")


def save_model(model, model_name: str, metadata: dict = None):
    """
    Save a PyTorch model's state dict and optional metadata.
    
    Args:
        model: PyTorch model (nn.Module)
        model_name: Name for the saved model file
        metadata: Optional dict of training metadata
    """
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    model_path = os.path.join(config.MODEL_SAVE_DIR, f"{model_name}.pth")
    
    save_dict = {
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
    }
    if metadata:
        save_dict["metadata"] = metadata

    torch.save(save_dict, model_path)
    logging.info(f"Model saved: {model_path}")
    return model_path


def load_model(model, model_name: str, device: torch.device = None):
    """
    Load a PyTorch model's state dict from file.
    
    Args:
        model: PyTorch model instance (nn.Module) with matching architecture
        model_name: Name of the saved model file
        device: Device to load the model to
    
    Returns:
        tuple: (model, metadata) — loaded model and any saved metadata
    """
    if device is None:
        device = config.DEVICE
    
    model_path = os.path.join(config.MODEL_SAVE_DIR, f"{model_name}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    metadata = checkpoint.get("metadata", {})
    logging.info(f"Model loaded: {model_path}")
    return model, metadata


def save_sklearn_model(model, model_name: str, metadata: dict = None):
    """
    Save a scikit-learn model using joblib.
    
    Args:
        model: Scikit-learn model/pipeline
        model_name: Name for the saved model file
        metadata: Optional dict of training metadata
    """
    import joblib
    
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    model_path = os.path.join(config.MODEL_SAVE_DIR, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    
    if metadata:
        meta_path = os.path.join(config.MODEL_SAVE_DIR, f"{model_name}_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
    
    logging.info(f"Sklearn model saved: {model_path}")
    return model_path


def load_sklearn_model(model_name: str):
    """
    Load a scikit-learn model from file.
    
    Args:
        model_name: Name of the saved model file
    
    Returns:
        model: Loaded scikit-learn model
    """
    import joblib
    
    model_path = os.path.join(config.MODEL_SAVE_DIR, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    logging.info(f"Sklearn model loaded: {model_path}")
    return model


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, description: str = ""):
        self.description = description
        self.elapsed = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        if self.description:
            logging.info(f"{self.description}: {self.elapsed:.2f}s")


class TrainingHistory:
    """Track and store training metrics across epochs."""
    
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.lr = []

    def update(self, train_loss, val_loss, train_acc, val_acc, lr=None):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        if lr is not None:
            self.lr.append(lr)

    def plot(self, title: str = "Training History", save_path: str = None):
        """Plot training curves (loss and accuracy)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        ax1.plot(self.train_loss, label="Train Loss", color="#FF6B6B", linewidth=2)
        ax1.plot(self.val_loss, label="Val Loss", color="#4ECDC4", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"{title} — Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(self.train_acc, label="Train Acc", color="#FF6B6B", linewidth=2)
        ax2.plot(self.val_acc, label="Val Acc", color="#4ECDC4", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title(f"{title} — Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logging.info(f"Training history plot saved: {save_path}")
        plt.close()

    def to_dict(self):
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
            "lr": self.lr,
        }


def count_parameters(model) -> int:
    """Count the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def print_separator(title: str = "", char: str = "=", width: int = 60):
    """Print a formatted separator line."""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(char * width)
