"""
models.py — Model Architectures
Smart Waste Segregation System

Implements:
1. Custom CNN (5 convolutional blocks) — Baseline
2. ResNet50 Transfer Learning — Fine-tuned pretrained model
3. ResNet50 Feature Extractor — For hybrid model & traditional ML
4. Traditional ML model wrappers (SVM, KNN, Decision Tree)
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import config


# ============================================================
# 1. CUSTOM CNN — BASELINE MODEL
# ============================================================

class ConvBlock(nn.Module):
    """
    Single convolutional block: Conv2D → BatchNorm → ReLU → MaxPool.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (filters)
        kernel_size: Convolution kernel size
        pool_size: Max pooling kernel size
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, pool_size: int = 2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class WasteCNN(nn.Module):
    """
    Custom CNN with 5 convolutional blocks for waste classification.
    
    Architecture:
        Block 1: Conv2D(3→32) → BatchNorm → ReLU → MaxPool(2,2)    → 112x112
        Block 2: Conv2D(32→64) → BatchNorm → ReLU → MaxPool(2,2)   → 56x56
        Block 3: Conv2D(64→128) → BatchNorm → ReLU → MaxPool(2,2)  → 28x28
        Block 4: Conv2D(128→256) → BatchNorm → ReLU → MaxPool(2,2) → 14x14
        Block 5: Conv2D(256→512) → BatchNorm → ReLU → MaxPool(2,2) → 7x7
        
        Flatten → Dense(512) → ReLU → Dropout(0.5) → Dense(num_classes) → Softmax
    
    Input: (batch, 3, 224, 224)
    Output: (batch, num_classes) — class probabilities
    
    Loss: Categorical Cross-Entropy
    Optimizer: Adam
    
    Args:
        num_classes: Number of output classes (default from config)
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super(WasteCNN, self).__init__()

        # 5 Convolutional blocks with increasing filters
        self.block1 = ConvBlock(3, 32)      # 224 → 112
        self.block2 = ConvBlock(32, 64)     # 112 → 56
        self.block3 = ConvBlock(64, 128)    # 56 → 28
        self.block4 = ConvBlock(128, 256)   # 28 → 14
        self.block5 = ConvBlock(256, 512)   # 14 → 7

        # Classification head
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 7 * 7, 512)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Convolutional blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        # Classification
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x  # Raw logits; use CrossEntropyLoss (includes softmax)

    def predict_proba(self, x):
        """Get class probabilities using softmax."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


# ============================================================
# 2. RESNET50 TRANSFER LEARNING
# ============================================================

class WasteResNet50(nn.Module):
    """
    ResNet50 with transfer learning for waste classification.
    
    Architecture:
        - Pretrained ResNet50 (ImageNet weights)
        - All base layers frozen initially
        - Custom classification head:
            Linear(2048→512) → ReLU → Dropout(0.3) → Linear(512→num_classes)
    
    Fine-tuning strategy:
        Phase 1: Train classification head only (freeze base)
        Phase 2: Unfreeze top N layers, fine-tune with low LR (1e-5)
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to load pretrained ImageNet weights
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        super(WasteResNet50, self).__init__()

        # Load pretrained ResNet50
        if pretrained:
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            logging.info("Loaded pretrained ResNet50 (ImageNet V2 weights)")
        else:
            self.backbone = models.resnet50(weights=None)
            logging.info("Initialized ResNet50 without pretrained weights")

        # Freeze all base layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Get the number of features from the original fc layer
        num_features = self.backbone.fc.in_features  # 2048

        # Replace the final classification layer with custom head
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

        # Ensure the new head is trainable
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

        logging.info(f"Custom classification head: {num_features} → 512 → {num_classes}")

    def forward(self, x):
        return self.backbone(x)

    def predict_proba(self, x):
        """Get class probabilities using softmax."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def freeze_base(self):
        """Freeze all backbone layers (except classification head)."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        logging.info("ResNet50 base layers frozen.")

    def unfreeze_top_layers(self, num_blocks: int = config.RESNET_UNFREEZE_LAYERS):
        """
        Unfreeze the top N residual blocks for fine-tuning.
        
        ResNet50 has 4 main blocks (layer1-4).
        Unfreezing layer4 and layer3 is typical for fine-tuning.
        
        Args:
            num_blocks: Number of top blocks to unfreeze (1-4)
        """
        # ResNet50 layer structure: layer1, layer2, layer3, layer4
        layers = ["layer4", "layer3", "layer2", "layer1"]
        layers_to_unfreeze = layers[:num_blocks]

        for name, param in self.backbone.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logging.info(f"Unfroze top {num_blocks} blocks. "
                     f"Trainable: {trainable:,}/{total:,} parameters "
                     f"({100 * trainable / total:.1f}%)")

    def get_optimizer_param_groups(self, head_lr: float, base_lr: float):
        """
        Get parameter groups with different learning rates for optimizer.
        
        Args:
            head_lr: Learning rate for classification head
            base_lr: Learning rate for unfrozen backbone layers
        
        Returns:
            list: Parameter groups for optimizer
        """
        head_params = list(self.backbone.fc.parameters())
        base_params = [p for name, p in self.backbone.named_parameters()
                      if "fc" not in name and p.requires_grad]

        param_groups = [
            {"params": head_params, "lr": head_lr},
        ]
        if base_params:
            param_groups.append({"params": base_params, "lr": base_lr})

        return param_groups


# ============================================================
# 3. RESNET50 FEATURE EXTRACTOR (for Hybrid Model)
# ============================================================

class ResNet50FeatureExtractor(nn.Module):
    """
    ResNet50 as a feature extractor using Global Average Pooling.
    
    Extracts 2048-dimensional feature vectors from images.
    Used by:
    - Hybrid Model (ResNet50 features → Chi-square → SVM)
    - Traditional ML models (features → SVM/KNN/DT)
    
    Args:
        pretrained: Whether to load pretrained ImageNet weights
    """

    def __init__(self, pretrained: bool = True):
        super(ResNet50FeatureExtractor, self).__init__()

        if pretrained:
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet = models.resnet50(weights=None)

        # Remove the final FC layer — keep everything up to avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze all layers (pure feature extraction)
        for param in self.features.parameters():
            param.requires_grad = False

        logging.info("ResNet50 Feature Extractor initialized (2048-d output)")

    def forward(self, x):
        """
        Extract features from input images.
        
        Args:
            x: Input tensor (batch, 3, 224, 224)
        
        Returns:
            torch.Tensor: Feature vectors (batch, 2048)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 2048, 1, 1) → (batch, 2048)
        return x


# ============================================================
# 4. TRADITIONAL ML MODELS
# ============================================================

def get_svm_model():
    """
    Create SVM classifier with cubic polynomial kernel.
    
    Configuration (as per paper):
        - Kernel: Polynomial (degree=3, cubic)
        - Multi-class: One-vs-One
        - Probability estimates enabled (for ROC-AUC)
    
    Returns:
        sklearn.svm.SVC: Configured SVM classifier
    """
    model = SVC(**config.SVM_PARAMS)
    logging.info(f"SVM initialized: kernel={config.SVM_PARAMS['kernel']}, "
                 f"degree={config.SVM_PARAMS['degree']}, "
                 f"strategy={config.SVM_PARAMS['decision_function_shape']}")
    return model


def get_knn_model():
    """
    Create KNN classifier.
    
    Configuration:
        - k=5 neighbors
        - Distance-weighted voting
        - Minkowski metric
    
    Returns:
        sklearn.neighbors.KNeighborsClassifier: Configured KNN classifier
    """
    model = KNeighborsClassifier(**config.KNN_PARAMS)
    logging.info(f"KNN initialized: k={config.KNN_PARAMS['n_neighbors']}, "
                 f"weights={config.KNN_PARAMS['weights']}")
    return model


def get_decision_tree_model():
    """
    Create Decision Tree classifier.
    
    Configuration:
        - Gini impurity criterion
        - Max depth: 20
        - Min samples split: 5
    
    Returns:
        sklearn.tree.DecisionTreeClassifier: Configured DT classifier
    """
    model = DecisionTreeClassifier(**config.DT_PARAMS)
    logging.info(f"Decision Tree initialized: criterion={config.DT_PARAMS['criterion']}, "
                 f"max_depth={config.DT_PARAMS['max_depth']}")
    return model


# ============================================================
# MODEL FACTORY
# ============================================================

def get_model(model_name: str, **kwargs):
    """
    Factory function to create model instances by name.
    
    Args:
        model_name: One of 'cnn', 'resnet50', 'feature_extractor',
                    'svm', 'knn', 'decision_tree'
        **kwargs: Additional arguments passed to model constructor
    
    Returns:
        Model instance
    """
    model_map = {
        "cnn": WasteCNN,
        "resnet50": WasteResNet50,
        "feature_extractor": ResNet50FeatureExtractor,
        "svm": get_svm_model,
        "knn": get_knn_model,
        "decision_tree": get_decision_tree_model,
    }

    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Choose from: {list(model_map.keys())}")

    model_class = model_map[model_name]

    if model_name in ("svm", "knn", "decision_tree"):
        return model_class()  # No kwargs for sklearn models
    else:
        return model_class(**kwargs)


def print_model_summary(model, model_name: str = ""):
    """Print a summary of model architecture and parameter count."""
    if isinstance(model, nn.Module):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable

        print(f"\n{'=' * 50}")
        print(f"Model: {model_name or model.__class__.__name__}")
        print(f"{'=' * 50}")
        print(f"  Total parameters:     {total:>12,}")
        print(f"  Trainable parameters: {trainable:>12,}")
        print(f"  Frozen parameters:    {frozen:>12,}")
        print(f"{'=' * 50}\n")
    else:
        print(f"\nModel: {model_name} — {type(model).__name__}")
        print(f"  Parameters: {model.get_params()}\n")


if __name__ == "__main__":
    """Quick test of all model architectures."""
    logging.basicConfig(level=logging.INFO)

    # Test Custom CNN
    cnn = WasteCNN()
    print_model_summary(cnn, "Custom CNN")
    dummy = torch.randn(2, 3, 224, 224)
    out = cnn(dummy)
    print(f"  CNN output shape: {out.shape}")  # (2, 6)

    # Test ResNet50
    resnet = WasteResNet50()
    print_model_summary(resnet, "ResNet50")
    out = resnet(dummy)
    print(f"  ResNet50 output shape: {out.shape}")  # (2, 6)

    # Test Feature Extractor
    extractor = ResNet50FeatureExtractor()
    features = extractor(dummy)
    print(f"  Feature extractor output shape: {features.shape}")  # (2, 2048)

    # Test Traditional ML
    svm = get_svm_model()
    knn = get_knn_model()
    dt = get_decision_tree_model()
    print("All models instantiated successfully!")
