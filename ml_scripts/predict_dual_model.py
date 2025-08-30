#!/usr/bin/env python3
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
    print("\n" + "="*60)
    print("DUAL MODEL PREDICTION RESULTS")
    print("="*60)
    print(f"Image: {os.path.basename(args.image_path)}")
    print(f"Ensemble Mode: {result['ensemble_mode']}")
    print(f"\nFinal Prediction: {result['prediction'].upper()}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Cancer Probability: {result['cancer_probability']:.1%}")
    
    print("\nIndividual Model Results:")
    for model, pred in result['individual_predictions'].items():
        print(f"  {model.upper()}: {pred['prediction']} "
              f"(confidence: {pred['confidence']:.1%})")
    
    if 'uncertainty' in result:
        print(f"\nUncertainty Level: {result['uncertainty']['level'].upper()}")
        print(f"Recommendation: {result['uncertainty']['recommendation']}")
    
    print("="*60)