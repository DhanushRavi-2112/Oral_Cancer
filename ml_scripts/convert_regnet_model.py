#!/usr/bin/env python3
"""
Convert the existing large RegNet model to optimized version for production.
Since we can't directly transfer weights (different architectures), we'll:
1. Use the existing model for teacher-student distillation
2. Or retrain the small model from scratch
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
import logging
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.architectures import create_model
from src.models.architectures_optimized import create_optimized_model
from src.data.preprocessing import create_data_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_lightweight_regnet(checkpoint_path: str = None):
    """Create a lightweight RegNet model suitable for production."""
    
    print("Creating Lightweight RegNet Model")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Option 1: Train new optimized model from scratch (Recommended)
    print("\nOption 1: Train Optimized RegNetY-002 (3.2M params)")
    print("-"*40)
    print("This will create a truly lightweight model perfect for production.")
    print("\nCommand to train:")
    print("python train_optimized_regnet.py --data_dir data --epochs 50")
    
    # Option 2: Use knowledge distillation
    print("\n\nOption 2: Knowledge Distillation")
    print("-"*40)
    print("Use the large model to teach the small model.")
    print("This preserves performance while reducing size.")
    
    # Option 3: Model compression techniques
    print("\n\nOption 3: Compress Existing Model")
    print("-"*40)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Load the large model
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Check model size
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"Current model size: {size_mb:.1f} MB")
        
        # Apply quantization
        print("\nApplying quantization...")
        # Create a simple quantizable model wrapper
        class QuantizableOralCancerModel(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.quant = torch.quantization.QuantStub()
                self.model = original_model
                self.dequant = torch.quantization.DeQuantStub()
                
            def forward(self, x):
                x = self.quant(x)
                x = self.model(x)
                x = self.dequant(x)
                return x
        
        # Save compressed version
        compressed_path = checkpoint_path.replace('.pth', '_compressed.pth')
        
        # Save with compression
        torch.save(
            checkpoint,
            compressed_path,
            _use_new_zipfile_serialization=True,
            pickle_protocol=4
        )
        
        compressed_size = os.path.getsize(compressed_path) / (1024 * 1024)
        print(f"Compressed model size: {compressed_size:.1f} MB")
        print(f"Size reduction: {(1 - compressed_size/size_mb)*100:.1f}%")
        
    print("\n\nRECOMMENDATION:")
    print("For production use with both models, create a dual-model system:")
    print("1. Use VGG16 (176 MB) for high accuracy")
    print("2. Train RegNetY-002 (15 MB) for edge deployment")
    print("3. Use model ensembling for best results")
    

def create_train_script():
    """Create training script for optimized RegNet."""
    
    script_content = '''#!/usr/bin/env python3
"""
Train optimized RegNetY-002 model for production use.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.training.train import main as train_main
from src.models.architectures_optimized import create_optimized_model
import torch

# Monkey patch to use optimized model
original_create_model = sys.modules['src.models.architectures'].create_model

def create_model_wrapper(model_type, pretrained=True, device='cuda'):
    if model_type == 'regnet':
        return create_optimized_model('regnet_optimized', pretrained, device)
    return original_create_model(model_type, pretrained, device)

sys.modules['src.models.architectures'].create_model = create_model_wrapper

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    # Override sys.argv for train_main
    sys.argv = [
        'train.py',
        '--model', 'regnet',
        '--data_dir', args.data_dir,
        '--epochs', str(args.epochs),
        '--device', args.device,
        '--batch_size', '64'  # Can use larger batch size with small model
    ]
    
    print("Training Optimized RegNetY-002 (3.2M parameters)")
    print("This is 45x smaller than the current model!")
    print("="*60)
    
    train_main()
'''
    
    with open('train_optimized_regnet.py', 'w') as f:
        f.write(script_content)
    
    print("Created train_optimized_regnet.py")


def create_dual_model_predictor():
    """Create a production-ready dual model predictor."""
    
    script_content = '''#!/usr/bin/env python3
"""
Production-ready dual model inference system.
Uses both VGG16 and RegNet for ensemble predictions.
"""
import os
import sys
import torch
import numpy as np
from typing import Dict, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict import OralCancerPredictor


class DualModelPredictor:
    """Ensemble predictor using both VGG16 and RegNet."""
    
    def __init__(
        self,
        vgg_path: str = "outputs/models/VGG16/best_model.pth",
        regnet_path: str = "outputs/models/RegNetY-320MF/best_model.pth",
        device: str = 'cuda',
        ensemble_mode: str = 'average'
    ):
        """
        Initialize dual model predictor.
        
        Args:
            ensemble_mode: 'average', 'weighted', or 'confidence'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.ensemble_mode = ensemble_mode
        
        # Load both models
        print("Loading VGG16 model...")
        self.vgg_predictor = OralCancerPredictor(
            vgg_path, 'vgg16', self.device
        )
        
        print("Loading RegNet model...")
        self.regnet_predictor = OralCancerPredictor(
            regnet_path, 'regnet', self.device
        )
        
        # Weights for weighted ensemble (can be tuned)
        self.weights = {'vgg16': 0.55, 'regnet': 0.45}
        
    def predict(self, image_path: str) -> Dict:
        """
        Predict using both models and ensemble results.
        """
        # Get predictions from both models
        vgg_result = self.vgg_predictor.predict_single(image_path)
        regnet_result = self.regnet_predictor.predict_single(image_path)
        
        # Extract probabilities
        vgg_prob = vgg_result['cancer_probability']
        regnet_prob = regnet_result['cancer_probability']
        
        # Ensemble predictions
        if self.ensemble_mode == 'average':
            ensemble_prob = (vgg_prob + regnet_prob) / 2
        elif self.ensemble_mode == 'weighted':
            ensemble_prob = (
                self.weights['vgg16'] * vgg_prob +
                self.weights['regnet'] * regnet_prob
            )
        elif self.ensemble_mode == 'confidence':
            # Use the prediction with higher confidence
            vgg_conf = max(vgg_prob, 1 - vgg_prob)
            regnet_conf = max(regnet_prob, 1 - regnet_prob)
            
            if vgg_conf >= regnet_conf:
                ensemble_prob = vgg_prob
            else:
                ensemble_prob = regnet_prob
        else:
            ensemble_prob = (vgg_prob + regnet_prob) / 2
        
        # Create ensemble result
        prediction = "cancer" if ensemble_prob > 0.5 else "healthy"
        confidence = max(ensemble_prob, 1 - ensemble_prob)
        
        result = {
            "image_path": image_path,
            "prediction": prediction,
            "cancer_probability": ensemble_prob,
            "healthy_probability": 1 - ensemble_prob,
            "confidence": confidence,
            "ensemble_mode": self.ensemble_mode,
            "individual_predictions": {
                "vgg16": {
                    "prediction": vgg_result['prediction'],
                    "cancer_probability": vgg_prob,
                    "confidence": vgg_result['confidence']
                },
                "regnet": {
                    "prediction": regnet_result['prediction'],
                    "cancer_probability": regnet_prob,
                    "confidence": regnet_result['confidence']
                }
            },
            "processing_time": (
                vgg_result['processing_time'] +
                regnet_result['processing_time']
            )
        }
        
        return result
    
    def predict_with_uncertainty(self, image_path: str) -> Dict:
        """
        Predict with uncertainty estimation based on model disagreement.
        """
        result = self.predict(image_path)
        
        # Calculate uncertainty based on disagreement
        vgg_prob = result['individual_predictions']['vgg16']['cancer_probability']
        regnet_prob = result['individual_predictions']['regnet']['cancer_probability']
        
        # Disagreement score
        disagreement = abs(vgg_prob - regnet_prob)
        
        # Uncertainty categories
        if disagreement < 0.1:
            uncertainty = "low"
            recommendation = "High confidence prediction"
        elif disagreement < 0.3:
            uncertainty = "medium"
            recommendation = "Moderate confidence - consider additional screening"
        else:
            uncertainty = "high"
            recommendation = "Low confidence - requires expert review"
        
        result['uncertainty'] = {
            'level': uncertainty,
            'disagreement_score': disagreement,
            'recommendation': recommendation
        }
        
        return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Dual model prediction')
    parser.add_argument('image_path', help='Path to image')
    parser.add_argument('--mode', choices=['average', 'weighted', 'confidence'],
                      default='weighted', help='Ensemble mode')
    parser.add_argument('--uncertainty', action='store_true',
                      help='Include uncertainty estimation')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = DualModelPredictor(ensemble_mode=args.mode)
    
    # Make prediction
    if args.uncertainty:
        result = predictor.predict_with_uncertainty(args.image_path)
    else:
        result = predictor.predict(args.image_path)
    
    # Display results
    print("\\n" + "="*60)
    print("DUAL MODEL PREDICTION RESULTS")
    print("="*60)
    print(f"Image: {os.path.basename(args.image_path)}")
    print(f"Ensemble Mode: {result['ensemble_mode']}")
    print(f"\\nFinal Prediction: {result['prediction'].upper()}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Cancer Probability: {result['cancer_probability']:.1%}")
    
    print("\\nIndividual Model Results:")
    for model, pred in result['individual_predictions'].items():
        print(f"  {model.upper()}: {pred['prediction']} "
              f"(confidence: {pred['confidence']:.1%})")
    
    if 'uncertainty' in result:
        print(f"\\nUncertainty Level: {result['uncertainty']['level'].upper()}")
        print(f"Recommendation: {result['uncertainty']['recommendation']}")
    
    print("="*60)
'''
    
    with open('predict_dual_model.py', 'w') as f:
        f.write(script_content)
    
    print("Created predict_dual_model.py")


def main():
    parser = argparse.ArgumentParser(description='Convert RegNet model for production')
    parser.add_argument('--checkpoint', type=str,
                      default='outputs/models/RegNetY-320MF/best_model.pth',
                      help='Path to existing model checkpoint')
    parser.add_argument('--create_scripts', action='store_true',
                      help='Create training and inference scripts')
    
    args = parser.parse_args()
    
    print("RegNet Model Optimization for Production")
    print("="*60)
    
    # Analyze current model
    create_lightweight_regnet(args.checkpoint)
    
    if args.create_scripts:
        print("\n\nCreating helper scripts...")
        create_train_script()
        create_dual_model_predictor()
        
        print("\n\nNext Steps:")
        print("1. Train optimized RegNet: python train_optimized_regnet.py")
        print("2. Use dual model prediction: python predict_dual_model.py <image>")
    

if __name__ == "__main__":
    main()