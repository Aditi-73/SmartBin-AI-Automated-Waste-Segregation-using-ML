"""
data_loader.py — Dataset Download, Preprocessing & Augmentation
Smart Waste Segregation System

Handles:
- TrashNet and Kaggle Garbage Classification dataset download
- Unified dataset creation with consistent class labels
- Image preprocessing (resize, normalize)
- Data augmentation (flip, rotation, brightness, noise)
- SMOTE for class imbalance handling
- DataLoader creation for training/validation/test splits
- K-Fold cross-validation splits
"""

import os
import shutil
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from PIL import Image

import config
from utils import set_seed


# ============================================================
# DATASET DOWNLOAD
# ============================================================

def download_trashnet():
    """
    Download the TrashNet dataset (6 classes, ~2527 images).
    Uses kagglehub if available, otherwise provides manual instructions.
    
    Returns:
        str: Path to the downloaded dataset directory
    """
    if os.path.exists(config.TRASHNET_DIR) and _count_images(config.TRASHNET_DIR) > 100:
        logging.info(f"TrashNet dataset already exists at {config.TRASHNET_DIR} "
                     f"({_count_images(config.TRASHNET_DIR)} images)")
        return config.TRASHNET_DIR

    logging.info("Downloading TrashNet dataset...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("asdasdasasdas/trashnet")
        logging.info(f"TrashNet downloaded to: {path}")

        # Find the actual image directories
        _organize_trashnet(path, config.TRASHNET_DIR)
        return config.TRASHNET_DIR

    except Exception as e:
        logging.warning(f"Auto-download failed: {e}")
        logging.info(
            "Please download TrashNet manually:\n"
            "  1. Go to https://www.kaggle.com/datasets/asdasdasasdas/trashnet\n"
            "  2. Download and extract to: {}\n"
            "  3. Ensure structure: trashnet/<class_name>/<images>".format(config.TRASHNET_DIR)
        )
        # Create placeholder structure
        os.makedirs(config.TRASHNET_DIR, exist_ok=True)
        return config.TRASHNET_DIR


def download_garbage_classification():
    """
    Download the Kaggle Garbage Classification dataset (12+ classes).
    
    Returns:
        str: Path to the downloaded dataset directory
    """
    if (os.path.exists(config.GARBAGE_CLASSIFICATION_DIR) and
            _count_images(config.GARBAGE_CLASSIFICATION_DIR) > 100):
        logging.info(f"Garbage Classification dataset already exists at "
                     f"{config.GARBAGE_CLASSIFICATION_DIR} "
                     f"({_count_images(config.GARBAGE_CLASSIFICATION_DIR)} images)")
        return config.GARBAGE_CLASSIFICATION_DIR

    logging.info("Downloading Garbage Classification dataset...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("mostafaabla/garbage-classification")
        logging.info(f"Garbage Classification downloaded to: {path}")

        _organize_garbage_classification(path, config.GARBAGE_CLASSIFICATION_DIR)
        return config.GARBAGE_CLASSIFICATION_DIR

    except Exception as e:
        logging.warning(f"Auto-download failed: {e}")
        logging.info(
            "Please download manually:\n"
            "  1. Go to https://www.kaggle.com/datasets/mostafaabla/garbage-classification\n"
            "  2. Download and extract to: {}\n".format(config.GARBAGE_CLASSIFICATION_DIR)
        )
        os.makedirs(config.GARBAGE_CLASSIFICATION_DIR, exist_ok=True)
        return config.GARBAGE_CLASSIFICATION_DIR


def _organize_trashnet(src_path: str, dst_path: str):
    """Organize TrashNet download into class-based directory structure."""
    os.makedirs(dst_path, exist_ok=True)

    # Search for image directories recursively
    for root, dirs, files in os.walk(src_path):
        # Look for directories that match class names
        dir_name = os.path.basename(root).lower()
        if dir_name in config.CLASSES:
            class_dir = os.path.join(dst_path, dir_name)
            os.makedirs(class_dir, exist_ok=True)
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    src_file = os.path.join(root, f)
                    dst_file = os.path.join(class_dir, f)
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_file, dst_file)

    # Verify
    total = _count_images(dst_path)
    logging.info(f"TrashNet organized: {total} images in {dst_path}")


def _organize_garbage_classification(src_path: str, dst_path: str):
    """Organize Garbage Classification download into class-based directory structure."""
    os.makedirs(dst_path, exist_ok=True)

    for root, dirs, files in os.walk(src_path):
        dir_name = os.path.basename(root).lower().replace(" ", "-")
        if dir_name and files:
            # Check if this looks like a class directory (has images)
            image_files = [f for f in files if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp"))]
            if image_files:
                class_dir = os.path.join(dst_path, dir_name)
                os.makedirs(class_dir, exist_ok=True)
                for f in image_files:
                    src_file = os.path.join(root, f)
                    dst_file = os.path.join(class_dir, f)
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_file, dst_file)

    total = _count_images(dst_path)
    logging.info(f"Garbage Classification organized: {total} images in {dst_path}")


def merge_datasets():
    """
    Merge TrashNet and Garbage Classification datasets into a unified dataset
    with consistent class labels (6 TrashNet classes).
    
    Returns:
        str: Path to unified dataset directory
    """
    if (os.path.exists(config.UNIFIED_DATA_DIR) and
            _count_images(config.UNIFIED_DATA_DIR) > 100):
        total = _count_images(config.UNIFIED_DATA_DIR)
        logging.info(f"Unified dataset already exists: {total} images")
        _print_class_distribution(config.UNIFIED_DATA_DIR)
        return config.UNIFIED_DATA_DIR

    logging.info("Merging datasets into unified directory...")
    os.makedirs(config.UNIFIED_DATA_DIR, exist_ok=True)

    # Create class directories
    for cls in config.CLASSES:
        os.makedirs(os.path.join(config.UNIFIED_DATA_DIR, cls), exist_ok=True)

    img_count = {cls: 0 for cls in config.CLASSES}

    # Copy from TrashNet
    if os.path.exists(config.TRASHNET_DIR):
        for cls in config.CLASSES:
            src_dir = os.path.join(config.TRASHNET_DIR, cls)
            if os.path.exists(src_dir):
                for f in os.listdir(src_dir):
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        src = os.path.join(src_dir, f)
                        dst = os.path.join(config.UNIFIED_DATA_DIR, cls,
                                           f"trashnet_{f}")
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
                            img_count[cls] += 1

    # Copy from Garbage Classification (with class mapping)
    if os.path.exists(config.GARBAGE_CLASSIFICATION_DIR):
        for gc_class, target_class in config.GARBAGE_CLASS_MAP.items():
            if target_class is None:
                continue
            src_dir = os.path.join(config.GARBAGE_CLASSIFICATION_DIR, gc_class)
            if os.path.exists(src_dir):
                for f in os.listdir(src_dir):
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        src = os.path.join(src_dir, f)
                        dst = os.path.join(config.UNIFIED_DATA_DIR, target_class,
                                           f"gc_{gc_class}_{f}")
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
                            img_count[target_class] += 1

    logging.info("Dataset merge complete:")
    for cls, count in img_count.items():
        logging.info(f"  {cls}: {count} new images")

    _print_class_distribution(config.UNIFIED_DATA_DIR)
    return config.UNIFIED_DATA_DIR


def _count_images(directory: str) -> int:
    """Count total image files in a directory tree."""
    count = 0
    if not os.path.exists(directory):
        return 0
    for root, _, files in os.walk(directory):
        count += sum(1 for f in files if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")))
    return count


def _print_class_distribution(data_dir: str):
    """Print the class distribution of a dataset directory."""
    logging.info("Class distribution:")
    total = 0
    for cls in sorted(os.listdir(data_dir)):
        cls_path = os.path.join(data_dir, cls)
        if os.path.isdir(cls_path):
            count = len([f for f in os.listdir(cls_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
            logging.info(f"  {cls}: {count}")
            total += count
    logging.info(f"  TOTAL: {total}")


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

def get_train_transforms():
    """
    Get training data transforms with augmentation.
    
    Augmentation pipeline (as per paper):
    - Resize to 256, random crop to 224
    - Random horizontal flip
    - Random rotation [-30°, +30°]
    - Brightness variation
    - Gaussian blur (approximates Gaussian noise)
    - Normalize using ImageNet statistics: x' = (x - μ) / σ
    
    Returns:
        torchvision.transforms.Compose: Training transform pipeline
    """
    return transforms.Compose([
        transforms.Resize((config.AUGMENTATION["resize_size"],
                          config.AUGMENTATION["resize_size"])),
        transforms.RandomCrop(config.AUGMENTATION["random_crop_size"]),
        transforms.RandomHorizontalFlip(p=config.AUGMENTATION["horizontal_flip_prob"]),
        transforms.RandomRotation(degrees=config.AUGMENTATION["rotation_degrees"]),
        transforms.ColorJitter(
            brightness=config.AUGMENTATION["brightness_variation"],
            contrast=0.1,
            saturation=0.1,
        ),
        transforms.GaussianBlur(
            kernel_size=config.AUGMENTATION["gaussian_blur_kernel"],
            sigma=(0.1, 2.0)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_val_transforms():
    """
    Get validation/test data transforms (no augmentation).
    
    Only resize, center crop, and normalize.
    
    Returns:
        torchvision.transforms.Compose: Validation transform pipeline
    """
    return transforms.Compose([
        transforms.Resize((config.AUGMENTATION["resize_size"],
                          config.AUGMENTATION["resize_size"])),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


# ============================================================
# CUSTOM DATASET
# ============================================================

class WasteDataset(Dataset):
    """
    Custom PyTorch Dataset for waste image classification.
    
    Loads images from a class-based directory structure:
        data_dir/<class_name>/<image_file>
    
    Args:
        data_dir: Root directory containing class subdirectories
        transform: Image transform pipeline
        classes: List of class names (default from config)
    """

    def __init__(self, data_dir: str, transform=None, classes=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = classes or config.CLASSES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []  # List of (image_path, label) tuples
        self.targets = []  # List of labels for stratified splitting

        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            if not os.path.exists(cls_dir):
                logging.warning(f"Class directory not found: {cls_dir}")
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    path = os.path.join(cls_dir, fname)
                    label = self.class_to_idx[cls]
                    self.samples.append((path, label))
                    self.targets.append(label)

        self.targets = np.array(self.targets)
        logging.info(f"WasteDataset: {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            # Return a blank image on error
            image = Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE))

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> dict:
        """Return a dict of class_name -> count."""
        counter = Counter(self.targets.tolist())
        return {self.classes[k]: v for k, v in sorted(counter.items())}


# ============================================================
# DATA SPLITTING
# ============================================================

def create_data_splits(dataset: WasteDataset):
    """
    Split dataset into train/validation/test sets (stratified).
    
    Split ratios from config: 70% train / 15% val / 15% test
    
    Args:
        dataset: WasteDataset instance
    
    Returns:
        tuple: (train_indices, val_indices, test_indices)
    """
    set_seed()
    indices = np.arange(len(dataset))
    labels = dataset.targets

    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        indices, test_size=config.TEST_RATIO,
        stratify=labels, random_state=config.RANDOM_SEED
    )

    # Second split: train vs val
    val_ratio_adjusted = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio_adjusted,
        stratify=labels[train_val_idx], random_state=config.RANDOM_SEED
    )

    logging.info(f"Data split — Train: {len(train_idx)}, Val: {len(val_idx)}, "
                 f"Test: {len(test_idx)}")
    return train_idx, val_idx, test_idx


def get_kfold_splits(dataset: WasteDataset, n_splits: int = config.NUM_FOLDS):
    """
    Generate K-Fold stratified cross-validation splits.
    
    Args:
        dataset: WasteDataset instance
        n_splits: Number of folds (default 5)
    
    Yields:
        tuple: (fold_num, train_indices, val_indices)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=config.RANDOM_SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(
            np.arange(len(dataset)), dataset.targets)):
        logging.info(f"Fold {fold + 1}/{n_splits} — "
                     f"Train: {len(train_idx)}, Val: {len(val_idx)}")
        yield fold, train_idx, val_idx


# ============================================================
# DATALOADERS
# ============================================================

def get_dataloaders(data_dir: str = None, batch_size: int = config.BATCH_SIZE):
    """
    Create train/validation/test DataLoaders.
    
    Args:
        data_dir: Path to dataset directory (default: unified data dir)
        batch_size: Batch size for DataLoaders
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, dataset)
    """
    if data_dir is None:
        data_dir = config.UNIFIED_DATA_DIR

    # Create full dataset with validation transforms (for splitting)
    full_dataset = WasteDataset(data_dir, transform=None)

    if len(full_dataset) == 0:
        raise ValueError(f"No images found in {data_dir}. "
                        "Please download datasets first.")

    # Get split indices
    train_idx, val_idx, test_idx = create_data_splits(full_dataset)

    # Create separate datasets with appropriate transforms
    train_dataset = WasteDataset(data_dir, transform=get_train_transforms())
    val_dataset = WasteDataset(data_dir, transform=get_val_transforms())
    test_dataset = WasteDataset(data_dir, transform=get_val_transforms())

    # Create subsets
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    test_subset = Subset(test_dataset, test_idx)

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )

    logging.info(f"DataLoaders created — "
                 f"Train: {len(train_subset)} ({len(train_loader)} batches), "
                 f"Val: {len(val_subset)} ({len(val_loader)} batches), "
                 f"Test: {len(test_subset)} ({len(test_loader)} batches)")

    return train_loader, val_loader, test_loader, full_dataset


def get_fold_dataloaders(dataset: WasteDataset, train_idx, val_idx,
                         batch_size: int = config.BATCH_SIZE):
    """
    Create DataLoaders for a specific cross-validation fold.
    
    Args:
        dataset: WasteDataset instance
        train_idx: Training indices for this fold
        val_idx: Validation indices for this fold
        batch_size: Batch size
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    data_dir = dataset.data_dir

    train_dataset = WasteDataset(data_dir, transform=get_train_transforms())
    val_dataset = WasteDataset(data_dir, transform=get_val_transforms())

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )

    return train_loader, val_loader


# ============================================================
# SMOTE FOR CLASS IMBALANCE
# ============================================================

def apply_smote(features: np.ndarray, labels: np.ndarray):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to handle
    class imbalance in extracted feature vectors.
    
    Used specifically for traditional ML pipelines (SVM, KNN, Decision Tree).
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Label array (n_samples,)
    
    Returns:
        tuple: (resampled_features, resampled_labels)
    """
    from imblearn.over_sampling import SMOTE

    logging.info(f"Applying SMOTE — Original distribution: {Counter(labels)}")

    smote = SMOTE(random_state=config.RANDOM_SEED, k_neighbors=3)
    features_resampled, labels_resampled = smote.fit_resample(features, labels)

    logging.info(f"After SMOTE — New distribution: {Counter(labels_resampled)}")
    logging.info(f"Samples: {len(labels)} → {len(labels_resampled)}")

    return features_resampled, labels_resampled


# ============================================================
# YOLO DATASET PREPARATION
# ============================================================

def prepare_yolo_dataset(data_dir: str = None):
    """
    Convert classification dataset to YOLO detection format.
    
    Creates synthetic bounding boxes (full-image) for single-object images.
    Generates the required directory structure and data.yaml config.
    
    Args:
        data_dir: Source dataset directory (default: unified)
    
    Returns:
        str: Path to data.yaml configuration file
    """
    if data_dir is None:
        data_dir = config.UNIFIED_DATA_DIR

    yolo_dir = config.YOLO_DATA_DIR
    yaml_path = os.path.join(yolo_dir, "data.yaml")

    if os.path.exists(yaml_path):
        logging.info(f"YOLO dataset already prepared at {yolo_dir}")
        return yaml_path

    logging.info("Preparing YOLO dataset format...")

    # Create directory structure
    for split in ["train", "val"]:
        os.makedirs(os.path.join(yolo_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, split, "labels"), exist_ok=True)

    # Load dataset and split
    full_dataset = WasteDataset(data_dir, transform=None)
    train_idx, val_idx, test_idx = create_data_splits(full_dataset)

    # Combine val and test for YOLO validation set
    val_combined = np.concatenate([val_idx, test_idx])

    def _copy_with_labels(indices, split_name):
        for idx in indices:
            img_path, label = full_dataset.samples[idx]
            fname = os.path.basename(img_path)
            name_no_ext = os.path.splitext(fname)[0]

            # Copy image
            dst_img = os.path.join(yolo_dir, split_name, "images", fname)
            if not os.path.exists(dst_img):
                shutil.copy2(img_path, dst_img)

            # Create YOLO label (full-image bounding box)
            # Format: class_id x_center y_center width height (all normalized 0-1)
            label_file = os.path.join(yolo_dir, split_name, "labels",
                                      f"{name_no_ext}.txt")
            with open(label_file, "w") as f:
                # Full image bounding box: center=(0.5, 0.5), size=(1.0, 1.0)
                f.write(f"{label} 0.5 0.5 1.0 1.0\n")

    _copy_with_labels(train_idx, "train")
    _copy_with_labels(val_combined, "val")

    # Create data.yaml
    yaml_content = (
        f"path: {yolo_dir}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"\n"
        f"nc: {config.NUM_CLASSES}\n"
        f"names: {config.CLASSES}\n"
    )
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    logging.info(f"YOLO dataset prepared at {yolo_dir}")
    logging.info(f"  Train: {len(train_idx)} images")
    logging.info(f"  Val: {len(val_combined)} images")
    logging.info(f"  Config: {yaml_path}")

    return yaml_path


# ============================================================
# DATASET INFORMATION
# ============================================================

def get_dataset_info(data_dir: str = None) -> dict:
    """
    Get comprehensive dataset statistics.
    
    Args:
        data_dir: Path to dataset directory
    
    Returns:
        dict: Dataset statistics (total, per_class, imbalance_ratio, etc.)
    """
    if data_dir is None:
        data_dir = config.UNIFIED_DATA_DIR

    info = {"data_dir": data_dir, "classes": {}, "total": 0}

    for cls in config.CLASSES:
        cls_dir = os.path.join(data_dir, cls)
        if os.path.exists(cls_dir):
            count = len([f for f in os.listdir(cls_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
        else:
            count = 0
        info["classes"][cls] = count
        info["total"] += count

    counts = list(info["classes"].values())
    if counts and min(counts) > 0:
        info["imbalance_ratio"] = max(counts) / min(counts)
    else:
        info["imbalance_ratio"] = float("inf")

    info["min_class"] = min(info["classes"], key=info["classes"].get)
    info["max_class"] = max(info["classes"], key=info["classes"].get)

    return info


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    print("Downloading datasets...")
    download_trashnet()
    download_garbage_classification()
    merge_datasets()
    info = get_dataset_info()
    print(f"\nDataset info: {info}")
