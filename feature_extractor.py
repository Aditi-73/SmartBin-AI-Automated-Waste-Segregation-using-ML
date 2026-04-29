"""
feature_extractor.py — Feature Extraction & Selection
Smart Waste Segregation System

Implements:
- ResNet50-based deep feature extraction (2048-d vectors)
- Chi-square feature selection (2048 → 128 features)
- Feature scaling (MinMaxScaler, StandardScaler)
- Full hybrid model pipeline (feature extraction → selection → SVM)
"""

import os
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

import config
from models import ResNet50FeatureExtractor


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(data_loader: DataLoader,
                     model: ResNet50FeatureExtractor = None,
                     device: torch.device = None,
                     save_path: str = None) -> tuple:
    """
    Extract 2048-dimensional feature vectors from images using ResNet50
    with Global Average Pooling.
    
    Args:
        data_loader: DataLoader providing (images, labels) batches
        model: Feature extractor model (creates new if None)
        device: Compute device
        save_path: Optional path to save extracted features (.npz)
    
    Returns:
        tuple: (features_array, labels_array)
            - features: np.ndarray of shape (n_samples, 2048)
            - labels: np.ndarray of shape (n_samples,)
    """
    if device is None:
        device = config.DEVICE

    if model is None:
        model = ResNet50FeatureExtractor(pretrained=True)

    model = model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    logging.info(f"Extracting features from {len(data_loader)} batches...")

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Feature Extraction"):
            images = images.to(device)
            features = model(images)                   # (batch, 2048)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    features_array = np.concatenate(all_features, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)

    logging.info(f"Extracted features: shape={features_array.shape}, "
                 f"labels={labels_array.shape}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, features=features_array, labels=labels_array)
        logging.info(f"Features saved to {save_path}")

    return features_array, labels_array


def extract_features_from_finetuned(data_loader: DataLoader,
                                     resnet_model,
                                     device: torch.device = None) -> tuple:
    """
    Extract features from a fine-tuned ResNet50 model (WasteResNet50).
    
    Uses the output just before the classification head (after avgpool),
    giving 2048-dimensional feature vectors from the fine-tuned backbone.
    
    Args:
        data_loader: DataLoader providing (images, labels) batches
        resnet_model: Fine-tuned WasteResNet50 instance
        device: Compute device
    
    Returns:
        tuple: (features_array, labels_array)
    """
    if device is None:
        device = config.DEVICE

    resnet_model = resnet_model.to(device)
    resnet_model.eval()

    # Create a hook to capture features before the FC layer
    features_list = []

    def hook_fn(module, input, output):
        # input to FC layer is the feature vector after avgpool
        features_list.append(input[0].detach().cpu().numpy())

    # Register hook on the first layer of the FC head
    # The backbone.fc is nn.Sequential, so hook on fc[0] (first Linear)
    hook = resnet_model.backbone.fc[0].register_forward_hook(hook_fn)

    all_labels = []

    logging.info("Extracting features from fine-tuned ResNet50...")

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Fine-tuned Feature Extraction"):
            images = images.to(device)
            _ = resnet_model(images)  # Forward pass triggers hook
            all_labels.append(labels.numpy())

    hook.remove()

    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)

    logging.info(f"Fine-tuned features: shape={features_array.shape}")
    return features_array, labels_array


# ============================================================
# FEATURE SCALING
# ============================================================

def scale_features_minmax(train_features: np.ndarray,
                          test_features: np.ndarray = None):
    """
    Apply MinMaxScaler to make features non-negative (required for chi-square).
    
    Args:
        train_features: Training feature matrix
        test_features: Optional test feature matrix
    
    Returns:
        tuple: (scaled_train, scaled_test, scaler)
    """
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_features)

    scaled_test = None
    if test_features is not None:
        scaled_test = scaler.transform(test_features)

    logging.info(f"MinMax scaling applied. Range: [0, 1]")
    return scaled_train, scaled_test, scaler


def scale_features_standard(train_features: np.ndarray,
                            test_features: np.ndarray = None):
    """
    Apply StandardScaler for zero-mean, unit-variance normalization.
    
    Args:
        train_features: Training feature matrix
        test_features: Optional test feature matrix
    
    Returns:
        tuple: (scaled_train, scaled_test, scaler)
    """
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_features)

    scaled_test = None
    if test_features is not None:
        scaled_test = scaler.transform(test_features)

    logging.info(f"Standard scaling applied. Mean→0, Std→1")
    return scaled_train, scaled_test, scaler


# ============================================================
# CHI-SQUARE FEATURE SELECTION
# ============================================================

def chi_square_selection(train_features: np.ndarray,
                         train_labels: np.ndarray,
                         test_features: np.ndarray = None,
                         k: int = config.SELECTED_FEATURES) -> tuple:
    """
    Apply Chi-square feature selection to reduce dimensionality.
    
    Selects the top-k features with highest chi-square scores,
    indicating strongest association with the target class.
    
    Pipeline:
        1. MinMaxScale features (chi-square requires non-negative values)
        2. Compute chi-square scores for each feature
        3. Select top-k features
    
    Args:
        train_features: Training feature matrix (n_samples, 2048)
        train_labels: Training labels (n_samples,)
        test_features: Optional test feature matrix
        k: Number of features to select (default 128)
    
    Returns:
        tuple: (selected_train, selected_test, selector, feature_indices)
    """
    logging.info(f"Chi-square feature selection: {train_features.shape[1]} → {k}")

    # Step 1: MinMax scaling (chi-square needs non-negative values)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_features)

    test_scaled = None
    if test_features is not None:
        test_scaled = scaler.transform(test_features)

    # Step 2: Chi-square feature selection
    selector = SelectKBest(score_func=chi2, k=k)
    selected_train = selector.fit_transform(train_scaled, train_labels)

    selected_test = None
    if test_features is not None:
        selected_test = selector.transform(test_scaled)

    # Get selected feature indices
    feature_indices = selector.get_support(indices=True)

    # Log top features by score
    scores = selector.scores_
    top_indices = np.argsort(scores)[-10:][::-1]
    logging.info(f"Top 10 feature indices by chi2 score: {top_indices}")
    logging.info(f"Top 10 scores: {scores[top_indices]}")
    logging.info(f"Selected features: {selected_train.shape[1]}")

    return selected_train, selected_test, selector, feature_indices


# ============================================================
# FULL HYBRID PIPELINE
# ============================================================

def hybrid_feature_pipeline(train_loader: DataLoader,
                            test_loader: DataLoader,
                            resnet_model=None,
                            use_finetuned: bool = True,
                            device: torch.device = None,
                            save_dir: str = None) -> dict:
    """
    Complete hybrid model feature pipeline:
    
    1. Extract 2048-d features from ResNet50 (GAP layer)
    2. Apply MinMax scaling
    3. Apply Chi-square feature selection (→ 128 features)
    4. Apply StandardScaler for SVM input normalization
    
    Args:
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        resnet_model: Optional fine-tuned ResNet50 model
        use_finetuned: Whether to use fine-tuned model for extraction
        device: Compute device
        save_dir: Directory to save intermediate features
    
    Returns:
        dict: {
            'train_features': selected & scaled training features,
            'test_features': selected & scaled test features,
            'train_labels': training labels,
            'test_labels': test labels,
            'selector': fitted SelectKBest,
            'scalers': dict of fitted scalers,
            'feature_indices': selected feature indices,
            'raw_train_features': raw 2048-d training features,
            'raw_test_features': raw 2048-d test features,
        }
    """
    if save_dir is None:
        save_dir = config.FEATURES_DIR
    os.makedirs(save_dir, exist_ok=True)

    logging.info("=" * 50)
    logging.info("HYBRID FEATURE PIPELINE")
    logging.info("=" * 50)

    # Step 1: Extract features
    logging.info("Step 1/4: Extracting ResNet50 features...")
    if use_finetuned and resnet_model is not None:
        train_features, train_labels = extract_features_from_finetuned(
            train_loader, resnet_model, device)
        test_features, test_labels = extract_features_from_finetuned(
            test_loader, resnet_model, device)
    else:
        extractor = ResNet50FeatureExtractor(pretrained=True)
        train_features, train_labels = extract_features(
            train_loader, extractor, device,
            save_path=os.path.join(save_dir, "train_features_raw.npz"))
        test_features, test_labels = extract_features(
            test_loader, extractor, device,
            save_path=os.path.join(save_dir, "test_features_raw.npz"))

    # Step 2: Chi-square feature selection (includes MinMax scaling)
    logging.info("Step 2/4: Chi-square feature selection...")
    selected_train, selected_test, selector, feature_indices = chi_square_selection(
        train_features, train_labels, test_features, k=config.SELECTED_FEATURES)

    # Step 3: Standard scaling for SVM input
    logging.info("Step 3/4: Standard scaling...")
    final_train, final_test, std_scaler = scale_features_standard(
        selected_train, selected_test)

    # Step 4: Save processed features
    logging.info("Step 4/4: Saving processed features...")
    np.savez(os.path.join(save_dir, "hybrid_train.npz"),
             features=final_train, labels=train_labels)
    np.savez(os.path.join(save_dir, "hybrid_test.npz"),
             features=final_test, labels=test_labels)

    result = {
        "train_features": final_train,
        "test_features": final_test,
        "train_labels": train_labels,
        "test_labels": test_labels,
        "selector": selector,
        "feature_indices": feature_indices,
        "std_scaler": std_scaler,
        "raw_train_features": train_features,
        "raw_test_features": test_features,
    }

    logging.info(f"Hybrid pipeline complete: "
                 f"{train_features.shape[1]}→{final_train.shape[1]} features")
    return result


def load_cached_features(save_dir: str = None) -> dict:
    """
    Load previously extracted and processed features from disk.
    
    Args:
        save_dir: Directory containing saved feature files
    
    Returns:
        dict: {'train_features', 'test_features', 'train_labels', 'test_labels'}
              or None if files don't exist
    """
    if save_dir is None:
        save_dir = config.FEATURES_DIR

    train_path = os.path.join(save_dir, "hybrid_train.npz")
    test_path = os.path.join(save_dir, "hybrid_test.npz")

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_data = np.load(train_path)
        test_data = np.load(test_path)
        logging.info("Loaded cached hybrid features.")
        return {
            "train_features": train_data["features"],
            "train_labels": train_data["labels"],
            "test_features": test_data["features"],
            "test_labels": test_data["labels"],
        }
    return None


if __name__ == "__main__":
    """Test feature extraction on dummy data."""
    logging.basicConfig(level=logging.INFO)

    # Create dummy data
    dummy_images = torch.randn(8, 3, 224, 224)
    dummy_labels = torch.randint(0, 6, (8,))

    # Test feature extractor
    extractor = ResNet50FeatureExtractor(pretrained=True)
    extractor.eval()
    with torch.no_grad():
        features = extractor(dummy_images)
    print(f"Feature shape: {features.shape}")  # (8, 2048)

    # Test chi-square selection
    feat_np = features.numpy()
    labels_np = dummy_labels.numpy()
    selected, _, _, indices = chi_square_selection(feat_np, labels_np, k=128)
    print(f"Selected features shape: {selected.shape}")  # (8, 128)
    print("Feature extraction pipeline test passed!")
