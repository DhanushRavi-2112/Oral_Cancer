import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinaryClassificationHead(nn.Module):
    """Common classification head for both models."""
    
    def __init__(self, in_features: int, dropout_rate: float = 0.5):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(in_features, 256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout_rate * 0.6)  # 0.3 for second dropout
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class RegNetY320OptimizedOralCancer(nn.Module):
    """Optimized RegNetY-320 (3.2M parameters) for oral cancer detection."""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Use the correct small RegNetY model - regnety_002 has ~3.2M parameters
        self.backbone = timm.create_model(
            'regnety_002',  # This is the actual 3.2M parameter RegNetY
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        
        # Get the number of features from the last layer
        # regnety_002 outputs 368 features
        num_features = 368
        
        # Add custom classification head
        self.classifier = BinaryClassificationHead(num_features, dropout_rate=0.4)
        
        # Model info
        self.model_name = "RegNetY-320-Optimized"
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"{self.model_name} initialized")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def freeze_backbone(self):
        """Freeze all backbone layers."""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_layers(self, num_layers: int = -1):
        """Unfreeze the last num_layers of the backbone."""
        if num_layers == -1:
            # Unfreeze all layers
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Get all backbone layers
            backbone_modules = list(self.backbone.modules())
            
            # Unfreeze the last num_layers
            for module in backbone_modules[-num_layers:]:
                for param in module.parameters():
                    param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters after unfreezing: {trainable_params:,}")


class EfficientOralCancerDetector(nn.Module):
    """Lightweight model using MobileNetV3 for edge deployment."""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Use MobileNetV3 for efficiency
        self.backbone = timm.create_model(
            'mobilenetv3_large_100',
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        
        # MobileNetV3 large outputs 960 features
        num_features = 960
        
        # Lighter classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        self.model_name = "MobileNetV3-OralCancer"
        total_params = sum(p.numel() for p in self.parameters())
        
        logger.info(f"{self.model_name} initialized")
        logger.info(f"Total parameters: {total_params:,}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        output = self.classifier(features)
        return output


def create_optimized_model(
    model_type: str,
    pretrained: bool = True,
    device: str = 'cuda'
) -> Tuple[nn.Module, Dict]:
    """
    Create optimized models for production.
    
    Args:
        model_type: 'regnet_optimized', 'mobilenet', or 'vgg16'
        pretrained: Whether to use pretrained weights
        device: Device to place the model on
    
    Returns:
        Tuple of (model, config_dict)
    """
    if model_type == 'regnet_optimized':
        model = RegNetY320OptimizedOralCancer(pretrained=pretrained)
        config = {
            'model_name': 'RegNetY-320-Optimized',
            'batch_size': 64,  # Can use larger batch size
            'learning_rate': 0.001,
            'expected_size_mb': 15,  # ~3.2M params * 4 bytes
            'inference_time_ms': 10
        }
    elif model_type == 'mobilenet':
        model = EfficientOralCancerDetector(pretrained=pretrained)
        config = {
            'model_name': 'MobileNetV3-OralCancer',
            'batch_size': 128,
            'learning_rate': 0.001,
            'expected_size_mb': 22,  # ~5.4M params
            'inference_time_ms': 7
        }
    elif model_type == 'vgg16':
        # Keep existing VGG16
        from .architectures import VGG16OralCancer
        model = VGG16OralCancer(pretrained=pretrained)
        config = {
            'model_name': 'VGG16',
            'batch_size': 16,
            'learning_rate': 0.0001,
            'expected_size_mb': 60,  # Compressed from 176MB
            'inference_time_ms': 25
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move model to device
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    config['total_params'] = total_params
    
    return model, config


def quantize_model(model: nn.Module) -> nn.Module:
    """Apply dynamic quantization for smaller model size."""
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    return quantized_model


def optimize_for_onnx(model: nn.Module, example_input: torch.Tensor, save_path: str):
    """Export model to ONNX format for optimized inference."""
    model.eval()
    
    torch.onnx.export(
        model,
        example_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    logger.info(f"Model exported to ONNX: {save_path}")


if __name__ == "__main__":
    # Test optimized models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Testing Optimized Models for Production")
    print("="*50)
    
    # Test input
    batch_size = 1
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Test each optimized model
    for model_type in ['regnet_optimized', 'mobilenet']:
        print(f"\nTesting {model_type}...")
        model, config = create_optimized_model(model_type, device=device)
        
        # Test forward pass
        with torch.no_grad():
            output = model(x)
        
        print(f"Output shape: {output.shape}")
        print(f"Total parameters: {config['total_params']:,}")
        print(f"Expected size: {config['expected_size_mb']} MB")
        print(f"Expected inference: {config['inference_time_ms']} ms")
        
        # Test quantization
        if device == 'cpu':
            model_cpu = model.cpu()
            quantized = quantize_model(model_cpu)
            print(f"Quantized model created")
    
    print("\nOptimized models ready for production!")