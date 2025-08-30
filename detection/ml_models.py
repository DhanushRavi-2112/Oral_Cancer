import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg16
import timm
from PIL import Image
import os
from django.conf import settings
import numpy as np
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

class VGG16Model(nn.Module):
    def __init__(self, num_classes=1):
        super(VGG16Model, self).__init__()
        # Don't load pretrained weights as we'll load our own
        self.backbone = vgg16(weights=None)
        
        # Replace classifier to match training architecture
        num_features = self.backbone.classifier[6].in_features
        self.backbone.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.backbone(x)

class RegNetModel(nn.Module):
    def __init__(self, num_classes=1):
        super(RegNetModel, self).__init__()
        # Don't load pretrained weights as we'll load our own
        self.backbone = timm.create_model('regnety_320', pretrained=False)
        # Get the actual number of features from the model
        num_features = self.backbone.head.fc.in_features
        self.backbone.head.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.backbone(x)

class OralCancerPredictor:
    def __init__(self):
        self.device = torch.device('cpu')  # Use CPU for web app
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize models as None (load on first use)
        self.vgg16_model = None
        self.regnet_model = None
        self.models_loaded = False
    
    def load_models(self):
        """Load trained models"""
        if self.models_loaded:
            return
            
        vgg16_path = os.path.join(settings.BASE_DIR, 'outputs', 'models', 'VGG16', 'best_model.pth')
        regnet_path = os.path.join(settings.BASE_DIR, 'outputs', 'models', 'RegNetY-320MF', 'best_model.pth')
        
        try:
            # Load VGG16
            if os.path.exists(vgg16_path):
                self.vgg16_model = VGG16Model()
                self.vgg16_model.load_state_dict(torch.load(vgg16_path, map_location=self.device, weights_only=False))
                self.vgg16_model.eval()
                print("VGG16 model loaded successfully")
            else:
                print(f"VGG16 model not found at {vgg16_path}")
        except Exception as e:
            print(f"Error loading VGG16 model: {e}")
        
        try:
            # Load RegNet
            if os.path.exists(regnet_path):
                self.regnet_model = RegNetModel()
                self.regnet_model.load_state_dict(torch.load(regnet_path, map_location=self.device, weights_only=False))
                self.regnet_model.eval()
                print("RegNet model loaded successfully")
            else:
                print(f"RegNet model not found at {regnet_path}")
        except Exception as e:
            print(f"Error loading RegNet model: {e}")
        
        self.models_loaded = True
    
    def predict_image(self, image_path):
        """Predict cancer from image using both models"""
        # Load models if not already loaded
        self.load_models()
        
        # Set deterministic mode
        torch.set_grad_enabled(False)
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Ensure consistent resizing
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Convert to tensor and normalize
            input_tensor = transforms.ToTensor()(image)
            input_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])(input_tensor)
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            
            predictions = {}
            
            # VGG16 prediction
            if self.vgg16_model is not None:
                self.vgg16_model.eval()  # Ensure eval mode
                vgg16_output = self.vgg16_model(input_tensor)
                vgg16_prob = vgg16_output.squeeze().item()
                
                # Ensure probability is between 0 and 1
                vgg16_prob = max(0.0, min(1.0, vgg16_prob))
                
                predictions['vgg16'] = {
                    'probability': vgg16_prob,
                    'prediction': 'Cancer' if vgg16_prob > 0.5 else 'Healthy',
                    'confidence': abs(vgg16_prob - 0.5) * 2
                }
                print(f"VGG16 prediction: {vgg16_prob:.4f}")
            
            # RegNet prediction
            if self.regnet_model is not None:
                self.regnet_model.eval()  # Ensure eval mode
                regnet_output = self.regnet_model(input_tensor)
                regnet_prob = regnet_output.squeeze().item()
                
                # Ensure probability is between 0 and 1
                regnet_prob = max(0.0, min(1.0, regnet_prob))
                
                predictions['regnet'] = {
                    'probability': regnet_prob,
                    'prediction': 'Cancer' if regnet_prob > 0.5 else 'Healthy',
                    'confidence': abs(regnet_prob - 0.5) * 2
                }
                print(f"RegNet prediction: {regnet_prob:.4f}")
            
            # Ensemble prediction
            if 'vgg16' in predictions and 'regnet' in predictions:
                ensemble_prob = (predictions['vgg16']['probability'] + predictions['regnet']['probability']) / 2
                predictions['ensemble'] = {
                    'probability': ensemble_prob,
                    'prediction': 'Cancer' if ensemble_prob > 0.5 else 'Healthy',
                    'confidence': abs(ensemble_prob - 0.5) * 2
                }
                print(f"Ensemble prediction: {ensemble_prob:.4f}")
            
            return predictions
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
        finally:
            torch.set_grad_enabled(True)

# Global predictor instance (lazy loading)
predictor = None

def get_predictor():
    """Get or create the global predictor instance"""
    global predictor
    if predictor is None:
        predictor = OralCancerPredictor()
    return predictor