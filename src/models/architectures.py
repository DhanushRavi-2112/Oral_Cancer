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


class RegNetY320OralCancer(nn.Module):
    """RegNetY-320MF model for oral cancer detection."""
    
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        
        # Load RegNetY-320MF from timm
        self.backbone = timm.create_model(
            'regnety_320',  # This is the 3.2M parameter version
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        
        # Get the number of features from the last layer dynamically
        # The actual RegNetY-320 model outputs 3712 features based on the error
        num_features = 3712
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Add custom classification head
        self.classifier = BinaryClassificationHead(num_features)
        
        # Model info
        self.model_name = "RegNetY-320MF"
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"{self.model_name} initialized")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def unfreeze_layers(self, num_layers: int = -1):
        """Unfreeze the last num_layers of the backbone."""
        if num_layers == -1:
            # Unfreeze all layers
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Get all backbone layers
            backbone_layers = list(self.backbone.children())
            
            # Unfreeze the last num_layers
            for layer in backbone_layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Unfroze layers. Trainable parameters: {trainable_params:,}")


class VGG16OralCancer(nn.Module):
    """VGG16 model for oral cancer detection."""
    
    def __init__(self, pretrained: bool = True, freeze_layers: int = 10):
        super().__init__()
        
        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=pretrained)
        
        # Extract features (all conv layers)
        self.features = vgg.features
        
        # Freeze early layers
        if freeze_layers > 0:
            # VGG16 features has 31 layers (conv, relu, pool)
            for i, layer in enumerate(self.features):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # Get the output size of features
        # VGG16 outputs 512 channels
        num_features = 512
        
        # Add custom classification head
        self.classifier = BinaryClassificationHead(num_features)
        
        # Model info
        self.model_name = "VGG16"
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"{self.model_name} initialized")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        output = self.classifier(features)
        return output
    
    def unfreeze_layers(self, num_layers: int = -1):
        """Progressive unfreezing for VGG16."""
        if num_layers == -1:
            # Unfreeze all layers
            for param in self.features.parameters():
                param.requires_grad = True
        else:
            # Get total number of layers
            total_layers = len(list(self.features.children()))
            
            # Unfreeze from layer (total - num_layers) onwards
            start_unfreeze = max(0, total_layers - num_layers)
            
            for i, layer in enumerate(self.features):
                if i >= start_unfreeze:
                    for param in layer.parameters():
                        param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Unfroze layers. Trainable parameters: {trainable_params:,}")


def create_model(
    model_type: str,
    pretrained: bool = True,
    device: str = 'cuda'
) -> Tuple[nn.Module, Dict]:
    """
    Create and return a model based on the specified type.
    
    Args:
        model_type: 'regnet' or 'vgg16'
        pretrained: Whether to use pretrained weights
        device: Device to place the model on
    
    Returns:
        Tuple of (model, config_dict)
    """
    if model_type.lower() == 'regnet':
        model = RegNetY320OralCancer(pretrained=pretrained)
        config = {
            'model_name': 'RegNetY-320MF',
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'weight_decay': 0.0001
        }
    elif model_type.lower() == 'vgg16':
        model = VGG16OralCancer(pretrained=pretrained)
        config = {
            'model_name': 'VGG16',
            'batch_size': 16,  # Smaller due to memory constraints
            'learning_rate': 0.0001,
            'optimizer': 'Adam',
            'weight_decay': 0.0001
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move model to device
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    config['total_params'] = total_params
    config['trainable_params'] = trainable_params
    
    return model, config


def test_models():
    """Test both model architectures."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    
    print("\nTesting RegNetY-320MF...")
    regnet_model, regnet_config = create_model('regnet', device=device)
    regnet_output = regnet_model(x)
    print(f"Output shape: {regnet_output.shape}")
    print(f"Config: {regnet_config}")
    
    print("\nTesting VGG16...")
    vgg_model, vgg_config = create_model('vgg16', device=device)
    vgg_output = vgg_model(x)
    print(f"Output shape: {vgg_output.shape}")
    print(f"Config: {vgg_config}")
    
    # Test progressive unfreezing
    print("\nTesting progressive unfreezing...")
    vgg_model.unfreeze_layers(6)
    regnet_model.unfreeze_layers(-1)
    
    print("\nModels tested successfully!")


if __name__ == "__main__":
    test_models()