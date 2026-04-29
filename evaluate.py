"""
evaluate.py — Evaluation Metrics & Visualization
Smart Waste Segregation System

Computes: Accuracy, Precision, Recall, F1, MCC, ROC-AUC, Confusion Matrix
Generates: Confusion matrix heatmaps, ROC curves, training history plots
"""

import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score,
                             confusion_matrix, classification_report, roc_curve, auc)
from sklearn.preprocessing import label_binarize

import config


def compute_metrics(y_true, y_pred, y_proba=None, average='weighted'):
    """
    Compute all evaluation metrics for a model.
    
    Returns: dict with accuracy, precision, recall, f1, mcc, roc_auc
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0) * 100,
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0) * 100,
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0) * 100,
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    # ROC-AUC (requires probability estimates)
    if y_proba is not None:
        try:
            y_bin = label_binarize(y_true, classes=list(range(config.NUM_CLASSES)))
            if y_proba.shape[1] == config.NUM_CLASSES:
                metrics["roc_auc"] = roc_auc_score(y_bin, y_proba, average='macro',
                                                    multi_class='ovr') * 100
            else:
                metrics["roc_auc"] = 0.0
        except Exception as e:
            logging.warning(f"ROC-AUC computation failed: {e}")
            metrics["roc_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0

    return metrics


def evaluate_dl_model(model, test_loader, device=None):
    """
    Evaluate a PyTorch deep learning model on test data.
    
    Returns: dict with metrics, predictions, probabilities
    """
    if device is None:
        device = config.DEVICE
    model.eval()
    model.to(device)

    all_preds, all_labels, all_probas = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probas = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probas.extend(probas.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probas)

    metrics = compute_metrics(y_true, y_pred, y_proba)
    
    report = classification_report(y_true, y_pred, target_names=config.CLASSES, zero_division=0)
    logging.info(f"\n{report}")

    return {"metrics": metrics, "y_true": y_true, "y_pred": y_pred, "y_proba": y_proba}


def evaluate_ml_model(model, test_features, test_labels, model_name="ML"):
    """
    Evaluate a scikit-learn model on test features.
    
    Returns: dict with metrics, predictions, probabilities
    """
    y_pred = model.predict(test_features)
    y_proba = model.predict_proba(test_features) if hasattr(model, 'predict_proba') else None

    metrics = compute_metrics(test_labels, y_pred, y_proba)
    logging.info(f"{model_name} — Acc: {metrics['accuracy']:.2f}%, "
                 f"F1: {metrics['f1_score']:.2f}%, MCC: {metrics['mcc']:.4f}")
    
    return {"metrics": metrics, "y_true": test_labels, "y_pred": y_pred, "y_proba": y_proba}


# ============================================================
# VISUALIZATION
# ============================================================

def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASSES, yticklabels=config.CLASSES)
    plt.title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(config.FIGURE_DIR, f"cm_{model_name.lower().replace(' ', '_')}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Confusion matrix saved: {save_path}")


def plot_roc_curves(y_true, y_proba, model_name="Model", save_path=None):
    """Plot per-class ROC curves with AUC scores."""
    if y_proba is None:
        logging.warning(f"No probabilities for ROC curve: {model_name}")
        return

    y_bin = label_binarize(y_true, classes=list(range(config.NUM_CLASSES)))
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, config.NUM_CLASSES))

    for i, (cls, color) in enumerate(zip(config.CLASSES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{cls} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves — {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.FIGURE_DIR, f"roc_{model_name.lower().replace(' ', '_')}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"ROC curves saved: {save_path}")


def plot_class_distribution(data_dir=None, save_path=None):
    """Plot dataset class distribution bar chart."""
    if data_dir is None:
        data_dir = config.UNIFIED_DATA_DIR
    
    counts = {}
    for cls in config.CLASSES:
        cls_dir = os.path.join(data_dir, cls)
        if os.path.exists(cls_dir):
            counts[cls] = len([f for f in os.listdir(cls_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        else:
            counts[cls] = 0

    plt.figure(figsize=(10, 6))
    bars = plt.bar(counts.keys(), counts.values(), color=plt.cm.Set2(np.linspace(0, 1, len(counts))))
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title('Dataset Class Distribution', fontsize=14, fontweight='bold')
    for bar, count in zip(bars, counts.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(count), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.FIGURE_DIR, "class_distribution.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def full_evaluation(model_name, y_true, y_pred, y_proba=None):
    """Run full evaluation: metrics + confusion matrix + ROC curves."""
    metrics = compute_metrics(y_true, y_pred, y_proba)
    plot_confusion_matrix(y_true, y_pred, model_name)
    if y_proba is not None:
        plot_roc_curves(y_true, y_proba, model_name)
    
    logging.info(f"\n{'='*50}\n{model_name} Evaluation Results\n{'='*50}")
    for k, v in metrics.items():
        logging.info(f"  {k}: {v:.4f}")
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Quick test with dummy data
    y_true = np.random.randint(0, 6, 100)
    y_pred = np.random.randint(0, 6, 100)
    y_proba = np.random.rand(100, 6)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    metrics = compute_metrics(y_true, y_pred, y_proba)
    print("Metrics:", metrics)
    print("Evaluation module test passed!")
