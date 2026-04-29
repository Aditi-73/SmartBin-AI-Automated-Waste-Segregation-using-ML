"""
yolo_train.py — YOLOv8 Training & Evaluation
Smart Waste Segregation System

Implements YOLOv8 for real-time waste detection:
- Dataset preparation (YOLO format)
- Model training
- Evaluation (IoU, Precision, Recall, F1, mAP)
"""

import os
import logging
import time
import numpy as np

import config
from data_loader import prepare_yolo_dataset
from utils import set_seed


def train_yolo(data_yaml: str = None, device=None):
    """
    Train YOLOv8 model for waste detection.
    
    Returns: (model, results, train_time)
    """
    from ultralytics import YOLO

    set_seed()
    if data_yaml is None:
        data_yaml = prepare_yolo_dataset()

    if device is None:
        device = "0" if config.CUDA_AVAILABLE else "cpu"

    logging.info(f"Training YOLOv8 — Model: {config.YOLO_MODEL}, Device: {device}")
    model = YOLO(config.YOLO_MODEL)

    start_time = time.time()
    results = model.train(
        data=data_yaml,
        epochs=config.YOLO_EPOCHS,
        imgsz=config.YOLO_IMG_SIZE,
        batch=config.YOLO_BATCH_SIZE,
        optimizer="Adam",
        device=device,
        project=os.path.join(config.OUTPUT_DIR, "yolo"),
        name="waste_detection",
        exist_ok=True,
        seed=config.RANDOM_SEED,
        verbose=True,
    )
    train_time = time.time() - start_time
    logging.info(f"YOLOv8 training complete in {train_time:.1f}s")
    return model, results, train_time


def evaluate_yolo(model=None, data_yaml: str = None):
    """
    Evaluate YOLOv8 model. Returns metrics dict.
    """
    from ultralytics import YOLO

    if model is None:
        best_path = os.path.join(config.OUTPUT_DIR, "yolo", "waste_detection", "weights", "best.pt")
        if os.path.exists(best_path):
            model = YOLO(best_path)
        else:
            logging.warning("No trained YOLO model found.")
            return None

    if data_yaml is None:
        data_yaml = os.path.join(config.YOLO_DATA_DIR, "data.yaml")

    logging.info("Evaluating YOLOv8...")
    metrics = model.val(data=data_yaml, verbose=True)

    results = {
        "mAP50": float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0.0,
        "mAP50-95": float(metrics.box.map) if hasattr(metrics.box, 'map') else 0.0,
        "precision": float(metrics.box.mp) if hasattr(metrics.box, 'mp') else 0.0,
        "recall": float(metrics.box.mr) if hasattr(metrics.box, 'mr') else 0.0,
    }
    results["f1"] = (2 * results["precision"] * results["recall"] /
                     (results["precision"] + results["recall"] + 1e-8))
    
    logging.info(f"YOLOv8 Results — mAP@0.5: {results['mAP50']:.4f}, "
                 f"P: {results['precision']:.4f}, R: {results['recall']:.4f}, "
                 f"F1: {results['f1']:.4f}")
    return results


def yolo_inference_time(model=None, data_yaml: str = None):
    """Measure YOLOv8 inference time per image."""
    from ultralytics import YOLO
    import glob

    if model is None:
        best_path = os.path.join(config.OUTPUT_DIR, "yolo", "waste_detection", "weights", "best.pt")
        if os.path.exists(best_path):
            model = YOLO(best_path)
        else:
            return 0.0

    val_images = os.path.join(config.YOLO_DATA_DIR, "val", "images")
    test_imgs = glob.glob(os.path.join(val_images, "*"))[:50]

    if not test_imgs:
        return 0.0

    start = time.time()
    for img in test_imgs:
        model.predict(img, verbose=False)
    elapsed = time.time() - start
    return (elapsed / len(test_imgs)) * 1000  # ms per image


def predict_single_image_yolo(image_path: str, model=None):
    """
    Run inference on a single image using the trained YOLOv8 model.
    """
    from ultralytics import YOLO

    if not os.path.exists(image_path):
        logging.error(f"Image not found: {image_path}")
        return

    if model is None:
        best_path = os.path.join(config.OUTPUT_DIR, "yolo", "waste_detection", "weights", "best.pt")
        if os.path.exists(best_path):
            logging.info(f"Loading trained YOLO model from {best_path}")
            model = YOLO(best_path)
        else:
            logging.warning("No trained YOLO model found. Have you trained it yet?")
            return

    logging.info(f"Running prediction on {image_path}")
    
    # Run prediction and save the annotated image
    results = model.predict(source=image_path, save=True, conf=0.5, 
                            project=os.path.join(config.OUTPUT_DIR, "yolo", "predictions"),
                            name="predict", exist_ok=True)
                            
    # Display the results
    result = results[0]
    logging.info(f"Detected {len(result.boxes)} objects.")
    
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        confidence = box.conf[0].item()
        logging.info(f"- Class: {class_name} (Confidence: {confidence:.2f})")
        
    logging.info(f"Annotated image saved in {result.save_dir}")
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("YOLOv8 module loaded successfully.")
