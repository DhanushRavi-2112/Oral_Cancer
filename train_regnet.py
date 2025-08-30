#!/usr/bin/env python3
"""
Script to train RegNet model with proper saving
"""
import subprocess
import os
import sys

# Ensure output directory exists
os.makedirs("outputs/models/RegNetY-320MF", exist_ok=True)

print("Starting RegNet training with proper model saving...")
print("="*60)

# Run training command
cmd = [
    sys.executable,
    "src/training/train.py",
    "--model", "regnet",
    "--data_dir", "data",
    "--epochs", "100",
    "--device", "cpu",
    "--save_dir", "outputs"
]

print(f"Running: {' '.join(cmd)}")
print("="*60)

try:
    result = subprocess.run(cmd, check=True)
    print("\nTraining completed successfully!")
    
    # Check if model was saved
    model_path = "outputs/models/RegNetY-320MF/best_model.pth"
    if os.path.exists(model_path):
        print(f"✓ Model saved at: {model_path}")
        print(f"✓ Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        print("\nYou can now test the model with:")
        print(f'python predict.py "C:\\Users\\user\\Desktop\\Oral_cancer\\orl_cnsr.jpg" --model {model_path} --model_type regnet --visualize')
    else:
        print("⚠ Warning: Model file not found at expected location")
        print("Checking for alternative locations...")
        
        # Search for any .pth files
        for root, dirs, files in os.walk("outputs"):
            for file in files:
                if file.endswith(".pth") and "regnet" in file.lower():
                    full_path = os.path.join(root, file)
                    print(f"Found: {full_path}")
                    
except subprocess.CalledProcessError as e:
    print(f"Training failed with error: {e}")
except KeyboardInterrupt:
    print("\nTraining interrupted by user")