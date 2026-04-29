"""
main.py — Master Orchestrator
Smart Waste Segregation System

Provides a CLI to run different parts of the pipeline:
- Data download & preparation
- Model training (CNN, ResNet50, Hybrid, YOLOv8)
- Evaluation & Comparison
"""

import os
import sys
import argparse
import logging

import config
from utils import setup_logging, create_output_dirs, get_device, set_seed
from data_loader import (download_trashnet, download_garbage_classification,
                         merge_datasets, get_dataloaders, get_dataset_info,
                         prepare_yolo_dataset)
from train import train_cnn, train_resnet50, train_traditional_ml, train_hybrid
from feature_extractor import extract_features, load_cached_features
from evaluate import full_evaluation
from compare import run_comparison
from yolo_train import train_yolo, evaluate_yolo, predict_single_image_yolo


def setup_environment():
    """Initialize directories, logging, and seed."""
    create_output_dirs()
    setup_logging()
    set_seed()
    logging.info("=" * 60)
    logging.info("SMART WASTE SEGREGATION SYSTEM INITIALIZED")
    logging.info("=" * 60)
    config.print_config()


def prepare_data():
    """Download and prepare all datasets."""
    logging.info("Preparing datasets...")
    download_trashnet()
    download_garbage_classification()
    merge_datasets()
    info = get_dataset_info()
    logging.info(f"Dataset ready. Total images: {info['total']}")
    return info


def run_training_pipeline():
    """Train all classification models (CNN, ResNet50, Traditional, Hybrid)."""
    logging.info("=" * 60)
    logging.info("STARTING TRAINING PIPELINE")
    logging.info("=" * 60)

    device = get_device()
    train_loader, val_loader, test_loader, _ = get_dataloaders()
    
    all_results = {}

    # 1. Custom CNN
    cnn_model, _, cnn_time = train_cnn(train_loader, val_loader, device)
    cnn_eval = full_evaluation("Custom CNN", *get_predictions(cnn_model, test_loader, device))
    all_results["Custom CNN"] = {"metrics": cnn_eval, "train_time": cnn_time, "inference_time": get_inference_time(cnn_model, test_loader, device)}

    # 2. ResNet50
    resnet_model, _, resnet_time = train_resnet50(train_loader, val_loader, device)
    resnet_eval = full_evaluation("ResNet50", *get_predictions(resnet_model, test_loader, device))
    all_results["ResNet50"] = {"metrics": resnet_eval, "train_time": resnet_time, "inference_time": get_inference_time(resnet_model, test_loader, device)}

    # 3. Hybrid Model (ResNet50 + SVM)
    hybrid_model, hybrid_pipeline, hybrid_time = train_hybrid(train_loader, test_loader, resnet_model, device)
    hybrid_eval = full_evaluation("Hybrid (ResNet50+SVM)", hybrid_pipeline["test_labels"], 
                                  hybrid_pipeline["predictions"], hybrid_pipeline["probabilities"])
    all_results["Hybrid (ResNet50+SVM)"] = {"metrics": hybrid_eval, "train_time": hybrid_time, 
                                            "inference_time": get_sklearn_inference_time(hybrid_model, hybrid_pipeline["test_features"])}

    # 4. Traditional ML
    ml_results = train_traditional_ml(hybrid_pipeline["raw_train_features"], hybrid_pipeline["train_labels"],
                                      hybrid_pipeline["raw_test_features"], hybrid_pipeline["test_labels"])
    for name, data in ml_results.items():
        eval_metrics = full_evaluation(name, hybrid_pipeline["test_labels"], data["predictions"], data["probabilities"])
        all_results[name] = {"metrics": eval_metrics, "train_time": data["train_time"],
                             "inference_time": get_sklearn_inference_time(data["model"], hybrid_pipeline["raw_test_features"])}

    # 5. Model Comparison
    run_comparison(all_results)
    return all_results


def run_yolo_pipeline():
    """Train and evaluate YOLOv8 model."""
    logging.info("=" * 60)
    logging.info("STARTING YOLO PIPELINE")
    logging.info("=" * 60)

    data_yaml = prepare_yolo_dataset()
    model, _, train_time = train_yolo(data_yaml)
    metrics = evaluate_yolo(model, data_yaml)
    
    # Store YOLO results in a compatible format for comparison later if needed
    logging.info(f"YOLO Training Time: {train_time:.1f}s")
    logging.info(f"YOLO Metrics: {metrics}")


def get_predictions(model, loader, device):
    """Helper to get predictions and true labels from a PyTorch model."""
    import torch
    import torch.nn.functional as F
    import numpy as np

    model.eval()
    all_preds, all_labels, all_probas = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probas = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probas.extend(probas.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds), np.array(all_probas)

def get_inference_time(model, loader, device):
    """Measure inference time (ms/image) for PyTorch model."""
    import time
    import torch
    model.eval()
    total_time, total_samples = 0.0, 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            start = time.perf_counter()
            _ = model(images)
            if device.type == 'cuda': torch.cuda.synchronize()
            total_time += time.perf_counter() - start
            total_samples += images.size(0)
    return (total_time / total_samples) * 1000

def get_sklearn_inference_time(model, features):
    """Measure inference time (ms/image) for Sklearn model."""
    import time
    start = time.perf_counter()
    _ = model.predict(features)
    elapsed = time.perf_counter() - start
    return (elapsed / len(features)) * 1000


def main():
    parser = argparse.ArgumentParser(description="Smart Waste Segregation System")
    parser.add_argument("--mode", type=str, default="full", 
                        choices=["data", "train", "yolo", "full", "predict_yolo"],
                        help="Mode to run (data=prepare only, train=classification models, yolo=train yolov8, full=everything, predict_yolo=run inference on single image)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an image to classify (used with --mode predict_yolo)")
    args = parser.parse_args()

    setup_environment()

    if args.mode in ["data", "full"]:
        prepare_data()
        
    if args.mode in ["train", "full"]:
        run_training_pipeline()
        
    if args.mode in ["yolo", "full"]:
        run_yolo_pipeline()

    if args.mode == "predict_yolo":
        if args.image is None:
            logging.error("Please provide an image path using --image when using --mode predict_yolo")
        else:
            predict_single_image_yolo(args.image)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user. Exiting...")
        sys.exit(0)
