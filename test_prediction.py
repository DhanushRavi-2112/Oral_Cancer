#!/usr/bin/env python3
"""
Simple script to test oral cancer prediction on an image
"""
import os
import sys

# Check if we have any trained models
model_dir = "outputs/models"
available_models = []

if os.path.exists(model_dir):
    for model_type in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_type, "best_model.pth")
        if os.path.exists(model_path):
            available_models.append((model_type, model_path))

if not available_models:
    print("No trained models found!")
    print("Please train a model first using:")
    print("  python src/training/train.py --model regnet --data_dir data")
    sys.exit(1)

print("Available trained models:")
for model_name, model_path in available_models:
    print(f"  - {model_name}: {model_path}")

# Get the first available model
model_name, model_path = available_models[0]

# Determine model type
if "VGG" in model_name.upper():
    model_type = "vgg16"
elif "REGNET" in model_name.upper():
    model_type = "regnet"
else:
    print(f"Unknown model type: {model_name}")
    sys.exit(1)

print(f"\nUsing {model_name} model ({model_type})")

# Test with a sample image
test_images = [
    "C:\\Users\\user\\Desktop\\Oral_cancer\\orl_cnsr.jpg",
    "dataset\\normal\\001.jpeg",
    "dataset\\Oral Cancer photos\\001.jpeg"
]

# Find first existing image
test_image = None
for img in test_images:
    if os.path.exists(img):
        test_image = img
        break

if not test_image:
    print("\nNo test image found. Please provide an image path.")
    print("\nExample command:")
    print(f'  python predict.py "YOUR_IMAGE_PATH" --model "{model_path}" --model_type {model_type} --visualize')
else:
    print(f"\nTesting with image: {test_image}")
    print("\nRun this command:")
    print(f'  python predict.py "{test_image}" --model "{model_path}" --model_type {model_type} --visualize --output results.json')