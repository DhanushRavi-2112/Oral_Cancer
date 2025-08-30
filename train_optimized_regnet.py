#!/usr/bin/env python3
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