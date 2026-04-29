"""
train.py — Training Pipeline
Smart Waste Segregation System

Training loops for all models:
- Custom CNN with cosine annealing LR
- ResNet50 two-phase transfer learning
- Traditional ML (SVM, KNN, Decision Tree)
- Hybrid model (ResNet50 + SVM)
- 5-fold cross-validation
"""

import os
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

import config
from models import WasteCNN, WasteResNet50, get_svm_model, get_knn_model, get_decision_tree_model
from feature_extractor import hybrid_feature_pipeline, extract_features
from data_loader import apply_smote, get_fold_dataloaders, get_kfold_splits
from utils import (save_model, save_sklearn_model, Timer, TrainingHistory,
                   count_parameters, set_seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train model for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


def validate(model, loader, criterion, device):
    """Validate model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


def train_cnn(train_loader, val_loader, device=None):
    """
    Train Custom CNN with cosine annealing LR scheduler.
    
    Returns: (model, history, train_time)
    """
    if device is None:
        device = config.DEVICE
    set_seed()

    model = WasteCNN(num_classes=config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.CNN_LEARNING_RATE,
                     betas=config.ADAM_BETAS, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.CNN_EPOCHS)

    history = TrainingHistory()
    best_val_acc, patience_counter = 0.0, 0

    logging.info(f"Training Custom CNN — {count_parameters(model):,} params")
    start_time = time.time()

    for epoch in range(config.CNN_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        history.update(train_loss, val_loss, train_acc, val_acc, optimizer.param_groups[0]['lr'])

        logging.info(f"Epoch {epoch+1}/{config.CNN_EPOCHS} — "
                     f"Train: {train_acc:.1f}% (loss={train_loss:.4f}) | "
                     f"Val: {val_acc:.1f}% (loss={val_loss:.4f})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, "cnn_best", {"epoch": epoch+1, "val_acc": val_acc})
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.CNN_EARLY_STOPPING_PATIENCE:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

    train_time = time.time() - start_time
    history.plot("Custom CNN", os.path.join(config.FIGURE_DIR, "cnn_history.png"))
    logging.info(f"CNN training complete — Best Val Acc: {best_val_acc:.2f}% in {train_time:.1f}s")
    return model, history, train_time


def train_resnet50(train_loader, val_loader, device=None):
    """
    Train ResNet50 with two-phase transfer learning:
    Phase 1: Train head only (frozen base)
    Phase 2: Fine-tune unfrozen top layers with low LR
    
    Returns: (model, history, train_time)
    """
    if device is None:
        device = config.DEVICE
    set_seed()

    model = WasteResNet50(num_classes=config.NUM_CLASSES, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    history = TrainingHistory()
    start_time = time.time()

    # Phase 1: Train classification head only
    logging.info("ResNet50 Phase 1: Training classification head...")
    optimizer = Adam(model.backbone.fc.parameters(), lr=config.RESNET_PHASE1_LR,
                     betas=config.ADAM_BETAS, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.RESNET_PHASE1_EPOCHS)

    for epoch in range(config.RESNET_PHASE1_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        history.update(train_loss, val_loss, train_acc, val_acc)
        logging.info(f"P1 Epoch {epoch+1}/{config.RESNET_PHASE1_EPOCHS} — "
                     f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")

    # Phase 2: Unfreeze top layers and fine-tune
    logging.info("ResNet50 Phase 2: Fine-tuning top layers...")
    model.unfreeze_top_layers(config.RESNET_UNFREEZE_LAYERS)
    param_groups = model.get_optimizer_param_groups(config.RESNET_PHASE1_LR * 0.1, config.RESNET_PHASE2_LR)
    optimizer = Adam(param_groups, betas=config.ADAM_BETAS, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.RESNET_PHASE2_EPOCHS)
    best_val_acc = 0.0

    for epoch in range(config.RESNET_PHASE2_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        history.update(train_loss, val_loss, train_acc, val_acc)
        logging.info(f"P2 Epoch {epoch+1}/{config.RESNET_PHASE2_EPOCHS} — "
                     f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, "resnet50_best", {"epoch": epoch+1, "val_acc": val_acc})

    train_time = time.time() - start_time
    history.plot("ResNet50 Transfer Learning", os.path.join(config.FIGURE_DIR, "resnet50_history.png"))
    logging.info(f"ResNet50 training complete — Best Val Acc: {best_val_acc:.2f}% in {train_time:.1f}s")
    return model, history, train_time


def train_traditional_ml(train_features, train_labels, test_features, test_labels, use_smote=True):
    """
    Train SVM, KNN, Decision Tree on extracted features.
    
    Returns: dict of {model_name: (model, train_time, predictions, probabilities)}
    """
    if use_smote:
        train_features, train_labels = apply_smote(train_features, train_labels)

    results = {}
    for name, get_model_fn in [("SVM", get_svm_model), ("KNN", get_knn_model),
                                ("Decision_Tree", get_decision_tree_model)]:
        logging.info(f"Training {name}...")
        model = get_model_fn()
        start = time.time()
        model.fit(train_features, train_labels)
        train_time = time.time() - start

        preds = model.predict(test_features)
        probas = model.predict_proba(test_features) if hasattr(model, 'predict_proba') else None
        save_sklearn_model(model, name.lower())

        # Cross-validation
        cv_scores = cross_val_score(get_model_fn(), train_features, train_labels,
                                    cv=config.NUM_FOLDS, scoring='accuracy', n_jobs=-1)
        logging.info(f"{name} — CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}, "
                     f"Train time: {train_time:.2f}s")

        results[name] = {
            "model": model, "train_time": train_time,
            "predictions": preds, "probabilities": probas,
            "cv_scores": cv_scores,
        }
    return results


def train_hybrid(train_loader, test_loader, resnet_model=None, device=None):
    """
    Train Hybrid Model: ResNet50 features → Chi-square → SVM.
    
    Returns: (svm_model, pipeline_result, train_time)
    """
    if device is None:
        device = config.DEVICE
    start = time.time()

    pipeline = hybrid_feature_pipeline(
        train_loader, test_loader, resnet_model, use_finetuned=(resnet_model is not None), device=device)

    svm = get_svm_model()
    train_features = pipeline["train_features"]
    train_labels = pipeline["train_labels"]

    # Apply SMOTE
    train_features_sm, train_labels_sm = apply_smote(train_features, train_labels)

    logging.info("Training Hybrid SVM on selected features...")
    svm.fit(train_features_sm, train_labels_sm)
    train_time = time.time() - start

    save_sklearn_model(svm, "hybrid_svm")

    preds = svm.predict(pipeline["test_features"])
    probas = svm.predict_proba(pipeline["test_features"])
    accuracy = np.mean(preds == pipeline["test_labels"]) * 100

    logging.info(f"Hybrid Model — Test Accuracy: {accuracy:.2f}%, Train time: {train_time:.2f}s")

    pipeline["predictions"] = preds
    pipeline["probabilities"] = probas
    return svm, pipeline, train_time


def measure_inference_time(model, loader, device=None, is_sklearn=False, features=None):
    """Measure per-sample inference time in milliseconds."""
    if device is None:
        device = config.DEVICE

    if is_sklearn and features is not None:
        start = time.time()
        _ = model.predict(features)
        elapsed = time.time() - start
        return (elapsed / len(features)) * 1000  # ms per sample

    model.eval()
    model.to(device)
    total_time, total_samples = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            start = time.time()
            _ = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            total_time += time.time() - start
            total_samples += images.size(0)
    return (total_time / total_samples) * 1000  # ms per sample


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Training module loaded successfully.")
    print(f"Device: {config.DEVICE}")
