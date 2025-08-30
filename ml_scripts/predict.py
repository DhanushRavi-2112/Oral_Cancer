#!/usr/bin/env python3
"""
Inference script for oral cancer detection.
Supports both single image and batch prediction.
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, List, Tuple, Union
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import cv2
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.architectures import create_model
from src.data.preprocessing import get_transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OralCancerPredictor:
    """Inference class for oral cancer detection."""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = 'regnet',
        device: str = 'cuda',
        confidence_threshold: float = 0.5
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            model_type: 'regnet' or 'vgg16'
            device: Device to run inference on
            confidence_threshold: Threshold for binary classification
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu' and device == 'cuda':
            logger.warning("CUDA not available, using CPU")
        
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Setup preprocessing
        self.transform = get_transforms(mode='test', model_type=model_type)
        
        logger.info(f"Predictor initialized with {model_type} model")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint."""
        # Create model
        model, _ = create_model(self.model_type, pretrained=False, device=self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        
        return model
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess a single image for inference."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        augmented = self.transform(image=image)
        image_tensor = augmented['image']
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def predict_single(self, image_path: str) -> Dict:
        """
        Predict on a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(image_tensor).squeeze()
            probability = torch.sigmoid(output).cpu().item()
        
        # Binary prediction
        prediction = "cancer" if probability > self.confidence_threshold else "healthy"
        
        processing_time = time.time() - start_time
        
        result = {
            "image_path": image_path,
            "prediction": prediction,
            "cancer_probability": probability,
            "healthy_probability": 1 - probability,
            "confidence": max(probability, 1 - probability),
            "processing_time": processing_time,
            "model_type": self.model_type
        }
        
        return result
    
    def predict_batch(self, image_paths: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Predict on multiple images.
        
        Args:
            image_paths: List of image file paths
            batch_size: Batch size for inference
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            
            # Preprocess batch
            for path in batch_paths:
                try:
                    tensor = self.preprocess_image(path)
                    batch_tensors.append(tensor)
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
                    results.append({
                        "image_path": path,
                        "error": str(e)
                    })
                    continue
            
            if not batch_tensors:
                continue
            
            # Stack tensors
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
            
            # Batch inference
            with torch.no_grad():
                outputs = self.model(batch_tensor).squeeze()
                if len(outputs.shape) == 0:  # Single item
                    outputs = outputs.unsqueeze(0)
                probabilities = torch.sigmoid(outputs).cpu().numpy()
            
            # Process results
            for j, (path, prob) in enumerate(zip(batch_paths[len(results) % batch_size:], probabilities)):
                prediction = "cancer" if prob > self.confidence_threshold else "healthy"
                
                results.append({
                    "image_path": path,
                    "prediction": prediction,
                    "cancer_probability": float(prob),
                    "healthy_probability": float(1 - prob),
                    "confidence": float(max(prob, 1 - prob)),
                    "model_type": self.model_type
                })
        
        return results
    
    def predict_directory(self, directory_path: str) -> List[Dict]:
        """
        Predict on all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            
        Returns:
            List of prediction dictionaries
        """
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_paths = []
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        
        if not image_paths:
            logger.warning(f"No images found in {directory_path}")
            return []
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        return self.predict_batch(image_paths)
    
    def visualize_prediction(
        self,
        image_path: str,
        result: Dict,
        save_path: str = None
    ):
        """Visualize prediction result on image."""
        import matplotlib.pyplot as plt
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(image)
        ax.axis('off')
        
        # Add prediction text
        prediction = result['prediction']
        confidence = result['confidence']
        
        # Color based on prediction
        color = 'red' if prediction == 'cancer' else 'green'
        
        # Add text box
        textstr = f"Prediction: {prediction.upper()}\nConfidence: {confidence:.2%}"
        props = dict(boxstyle='round', facecolor=color, alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=14,
               verticalalignment='top', bbox=props, color='white', weight='bold')
        
        # Add probability bar
        ax2 = fig.add_axes([0.15, 0.05, 0.7, 0.03])
        cancer_prob = result['cancer_probability']
        ax2.barh(0, cancer_prob, color='red', alpha=0.8, label='Cancer')
        ax2.barh(0, 1-cancer_prob, left=cancer_prob, color='green', alpha=0.8, label='Healthy')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax2.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax2.set_yticks([])
        ax2.set_xlabel('Probability', fontsize=10)
        
        plt.suptitle(f"Oral Cancer Detection - {os.path.basename(image_path)}", fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Oral cancer detection inference')
    parser.add_argument('input', type=str,
                       help='Path to image file, directory, or list of images')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='regnet',
                       choices=['regnet', 'vgg16'],
                       help='Model architecture type')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for cancer detection')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize predictions')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results (JSON format)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing multiple images')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = OralCancerPredictor(
        model_path=args.model,
        model_type=args.model_type,
        device=args.device,
        confidence_threshold=args.threshold
    )
    
    # Determine input type and predict
    if os.path.isfile(args.input):
        # Single image
        logger.info(f"Processing single image: {args.input}")
        result = predictor.predict_single(args.input)
        results = [result]
        
        # Print result
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"Image: {os.path.basename(args.input)}")
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Cancer Probability: {result['cancer_probability']:.2%}")
        print(f"Processing Time: {result['processing_time']:.3f}s")
        print("="*60)
        
        # Visualize if requested
        if args.visualize:
            save_path = args.output.replace('.json', '_viz.png') if args.output else None
            predictor.visualize_prediction(args.input, result, save_path)
            
    elif os.path.isdir(args.input):
        # Directory of images
        logger.info(f"Processing directory: {args.input}")
        results = predictor.predict_directory(args.input)
        
        # Print summary
        if results:
            cancer_count = sum(1 for r in results if r.get('prediction') == 'cancer')
            healthy_count = sum(1 for r in results if r.get('prediction') == 'healthy')
            error_count = sum(1 for r in results if 'error' in r)
            
            print("\n" + "="*60)
            print("BATCH PREDICTION SUMMARY")
            print("="*60)
            print(f"Total Images: {len(results)}")
            print(f"Cancer Detected: {cancer_count} ({cancer_count/len(results)*100:.1f}%)")
            print(f"Healthy: {healthy_count} ({healthy_count/len(results)*100:.1f}%)")
            if error_count:
                print(f"Errors: {error_count}")
            print("="*60)
            
            # Show individual results
            print("\nIndividual Results:")
            for result in results[:10]:  # Show first 10
                if 'error' in result:
                    print(f"  {os.path.basename(result['image_path'])}: ERROR - {result['error']}")
                else:
                    print(f"  {os.path.basename(result['image_path'])}: "
                          f"{result['prediction'].upper()} "
                          f"(confidence: {result['confidence']:.2%})")
            
            if len(results) > 10:
                print(f"  ... and {len(results) - 10} more")
    else:
        # List of images
        with open(args.input, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Processing {len(image_paths)} images from list")
        results = predictor.predict_batch(image_paths, batch_size=args.batch_size)
    
    # Save results if output specified
    if args.output and results:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()