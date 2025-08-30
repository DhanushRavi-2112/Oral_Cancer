#!/usr/bin/env python3
"""
Quick test script to compare both models
"""
import subprocess
import sys
import os

print("Testing Both Trained Models")
print("="*60)

# Test images
test_images = [
    ("Cancer Sample", "dataset\\Oral Cancer photos\\001.jpeg"),
    ("Healthy Sample", "dataset\\normal\\001.jpeg"),
]

# Models
models = [
    ("VGG16", "outputs/models/VGG16/best_model.pth", "vgg16"),
    ("RegNetY-320", "outputs/models/RegNetY-320MF/best_model.pth", "regnet"),
]

# Check if models exist
for model_name, model_path, _ in models:
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024*1024)
        print(f"✓ {model_name}: {size_mb:.1f} MB")
    else:
        print(f"✗ {model_name}: Not found")

print("\n" + "="*60)
print("Testing predictions on sample images:")
print("="*60)

# Test each model
for model_name, model_path, model_type in models:
    if not os.path.exists(model_path):
        continue
        
    print(f"\n{model_name} Results:")
    print("-"*40)
    
    for image_name, image_path in test_images:
        if os.path.exists(image_path):
            cmd = [
                sys.executable,
                "predict.py",
                image_path,
                "--model", model_path,
                "--model_type", model_type,
                "--output", f"test_{model_type}_{image_name.replace(' ', '_')}.json"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                # Parse output for results
                output = result.stdout
                if "Prediction: CANCER" in output:
                    prediction = "CANCER"
                elif "Prediction: HEALTHY" in output:
                    prediction = "HEALTHY"
                else:
                    prediction = "ERROR"
                    
                if "Confidence:" in output:
                    for line in output.split('\n'):
                        if "Confidence:" in line:
                            confidence = line.split("Confidence:")[1].strip()
                            break
                else:
                    confidence = "N/A"
                    
                print(f"  {image_name}: {prediction} (Confidence: {confidence})")
                
                if result.stderr:
                    print(f"    Warning: {result.stderr.split('WARNING:')[1].split('\\n')[0] if 'WARNING:' in result.stderr else ''}")
                    
            except subprocess.TimeoutExpired:
                print(f"  {image_name}: TIMEOUT")
            except Exception as e:
                print(f"  {image_name}: ERROR - {str(e)}")

print("\n" + "="*60)
print("To run full comparison analysis:")
print("python src/evaluation/compare_models.py --data_dir data --device cpu")
print("="*60)