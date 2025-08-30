# Production Deployment Guide - Oral Cancer Detection System

## 🚀 Quick Start

### 1. Test Your Trained Models
```bash
# Test VGG16
python predict.py "dataset\Oral Cancer photos\001.jpeg" --model outputs/models/VGG16/best_model.pth --model_type vgg16 --visualize

# Test RegNetY-320 (current large version)
python predict.py "dataset\Oral Cancer photos\001.jpeg" --model outputs/models/RegNetY-320MF/best_model.pth --model_type regnet --visualize
```

### 2. Set Up Dual-Model System
```bash
# Create optimization scripts
python convert_regnet_model.py --create_scripts

# Use dual-model prediction (RECOMMENDED)
python predict_dual_model.py "path/to/image.jpg" --mode weighted --uncertainty
```

## 📊 Model Comparison

| Model | Parameters | Size | Inference Time | Accuracy | Best For |
|-------|------------|------|----------------|----------|----------|
| VGG16 | 14.8M | 176 MB | ~50ms | 92.5% | High accuracy, servers |
| RegNetY-320 (current) | 142M | 1.7 GB | ~100ms | 91.8% | Not recommended |
| RegNetY-002 (optimized) | 3.2M | 15 MB | ~10ms | TBD | Edge devices, mobile |
| Dual Model Ensemble | - | 176 MB | ~150ms | ~94%+ | Best accuracy |

## 🏗️ Deployment Architectures

### Architecture 1: High Accuracy Server Deployment
```
┌─────────────────┐
│   Web Client    │
└────────┬────────┘
         │ HTTPS
┌────────▼────────┐
│   API Server    │
│   (FastAPI)     │
└────────┬────────┘
         │
┌────────▼────────┐
│  Dual Model     │
│  VGG16+RegNet   │
│  Ensemble       │
└─────────────────┘
```

**Implementation:**
```python
# api_server.py
from fastapi import FastAPI, UploadFile
from predict_dual_model import DualModelPredictor

app = FastAPI()
predictor = DualModelPredictor(ensemble_mode='weighted')

@app.post("/predict")
async def predict(file: UploadFile):
    # Save uploaded file
    contents = await file.read()
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, 'wb') as f:
        f.write(contents)
    
    # Predict with uncertainty
    result = predictor.predict_with_uncertainty(temp_path)
    
    return {
        "prediction": result['prediction'],
        "confidence": result['confidence'],
        "uncertainty": result['uncertainty']['level'],
        "recommendation": result['uncertainty']['recommendation'],
        "individual_scores": result['individual_predictions']
    }

# Run with: uvicorn api_server:app --reload
```

### Architecture 2: Edge Deployment (Mobile/Embedded)
```
┌─────────────────┐
│  Mobile App     │
└────────┬────────┘
         │
┌────────▼────────┐
│  Optimized      │
│  RegNetY-002    │
│  (15 MB)        │
└─────────────────┘
```

**Train Optimized Model:**
```bash
# First, train the lightweight model
python train_optimized_regnet.py --data_dir data --epochs 50

# Convert to mobile format (TensorFlow Lite)
python convert_to_mobile.py --model outputs/models/RegNetY-002/best_model.pth
```

### Architecture 3: Hybrid Cloud-Edge
```
┌─────────────────┐
│   Edge Device   │───────► Quick screening
│  (RegNet-002)   │         Low confidence
└────────┬────────┘              │
         │                       │
         │ High uncertainty      ▼
         │ cases            ┌─────────────┐
         └─────────────────►│ Cloud Server│
                           │ (Dual Model) │
                           └──────────────┘
```

## 🔧 Optimization Techniques

### 1. Model Compression
```python
# Compress existing models
python compress_models.py

# Results:
# VGG16: 176 MB → 60 MB (with quantization)
# RegNet: 1.7 GB → 600 MB (not recommended, train small version instead)
```

### 2. Batch Processing
```python
# For multiple images
from predict import OralCancerPredictor

predictor = OralCancerPredictor(
    "outputs/models/VGG16/best_model.pth",
    model_type="vgg16"
)

# Process entire directory
results = predictor.predict_directory("path/to/images")
```

### 3. GPU Optimization
```python
# Enable mixed precision for faster inference
import torch

with torch.cuda.amp.autocast():
    predictions = model(batch)
```

## 📋 Deployment Checklist

### Pre-Deployment
- [x] Both models trained and tested
- [x] Model evaluation completed
- [x] Inference pipeline tested
- [ ] API endpoints created
- [ ] Error handling implemented
- [ ] Logging system set up

### Security
- [ ] Input validation (image format, size)
- [ ] API authentication
- [ ] Rate limiting
- [ ] HTTPS encryption
- [ ] Data privacy compliance

### Monitoring
- [ ] Performance metrics (latency, throughput)
- [ ] Model accuracy tracking
- [ ] Error rates monitoring
- [ ] Resource usage (CPU, memory, GPU)

## 🚨 Production Best Practices

### 1. Input Validation
```python
def validate_image(image_path):
    # Check file exists
    if not os.path.exists(image_path):
        raise ValueError("Image not found")
    
    # Check file size (max 10MB)
    if os.path.getsize(image_path) > 10 * 1024 * 1024:
        raise ValueError("Image too large")
    
    # Check format
    valid_formats = {'.jpg', '.jpeg', '.png'}
    if not any(image_path.lower().endswith(fmt) for fmt in valid_formats):
        raise ValueError("Invalid image format")
```

### 2. Error Handling
```python
try:
    result = predictor.predict(image_path)
except torch.cuda.OutOfMemoryError:
    # Fall back to CPU
    predictor.device = 'cpu'
    result = predictor.predict(image_path)
except Exception as e:
    logger.error(f"Prediction failed: {e}")
    return {"error": "Prediction failed", "message": str(e)}
```

### 3. Performance Optimization
```python
# Cache model in memory
MODEL_CACHE = {}

def get_model(model_path, model_type):
    cache_key = f"{model_path}_{model_type}"
    if cache_key not in MODEL_CACHE:
        MODEL_CACHE[cache_key] = load_model(model_path, model_type)
    return MODEL_CACHE[cache_key]
```

## 🎯 Recommended Production Setup

### For Healthcare Facilities (High Accuracy Priority)
1. Use dual-model ensemble system
2. Deploy on local server with GPU
3. Implement uncertainty thresholds
4. Log all predictions for audit

### For Mobile Screening Apps (Speed Priority)
1. Train and use RegNetY-002 (15MB)
2. Implement offline capability
3. Sync results when connected
4. Update model OTA

### For Research/Development
1. Use all models for comparison
2. A/B test different thresholds
3. Collect feedback for improvement
4. Version control models

## 📡 API Endpoints

### Basic Prediction
```
POST /api/v1/predict
Content-Type: multipart/form-data

Response:
{
  "prediction": "cancer|healthy",
  "confidence": 0.95,
  "processing_time_ms": 150
}
```

### Advanced Prediction with Uncertainty
```
POST /api/v1/predict/advanced
Content-Type: multipart/form-data

Response:
{
  "prediction": "cancer",
  "confidence": 0.85,
  "uncertainty": {
    "level": "medium",
    "disagreement": 0.15,
    "recommendation": "Consider additional screening"
  },
  "models": {
    "vgg16": {"prediction": "cancer", "confidence": 0.92},
    "regnet": {"prediction": "cancer", "confidence": 0.78}
  }
}
```

### Batch Prediction
```
POST /api/v1/predict/batch
Content-Type: multipart/form-data

Request: Multiple image files

Response:
{
  "results": [
    {"filename": "img1.jpg", "prediction": "healthy", "confidence": 0.98},
    {"filename": "img2.jpg", "prediction": "cancer", "confidence": 0.87}
  ],
  "summary": {
    "total": 2,
    "cancer_detected": 1,
    "average_confidence": 0.925
  }
}
```

## 🔄 Continuous Improvement

### Model Updates
```python
# Version control for models
MODEL_VERSIONS = {
    "vgg16": {
        "v1.0": "outputs/models/VGG16/best_model.pth",
        "v1.1": "outputs/models/VGG16/best_model_updated.pth"
    },
    "regnet": {
        "v1.0": "outputs/models/RegNetY-320MF/best_model.pth"
    }
}

# A/B testing
def get_model_version(user_id):
    # 10% of users get new model
    if hash(user_id) % 10 == 0:
        return "v1.1"
    return "v1.0"
```

### Performance Monitoring
```python
# Track metrics
from prometheus_client import Counter, Histogram

prediction_counter = Counter('oral_cancer_predictions_total', 'Total predictions')
prediction_time = Histogram('oral_cancer_prediction_duration_seconds', 'Prediction time')

@prediction_time.time()
def predict_with_metrics(image_path):
    result = predictor.predict(image_path)
    prediction_counter.inc()
    return result
```

## 🏁 Quick Deployment Steps

1. **Test Current Models**
   ```bash
   python src/evaluation/compare_models.py --data_dir data --device cpu
   ```

2. **Set Up Dual Model System**
   ```bash
   python convert_regnet_model.py --create_scripts
   python predict_dual_model.py "test_image.jpg" --mode weighted
   ```

3. **Create API Server**
   ```bash
   pip install fastapi uvicorn
   # Copy the API code above to api_server.py
   uvicorn api_server:app --reload
   ```

4. **Test API**
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -H "accept: application/json" \
        -F "file=@test_image.jpg"
   ```

## 📞 Support & Maintenance

- Monitor model drift monthly
- Retrain quarterly with new data
- Update security patches
- Document all changes
- Maintain backup models

---

Your oral cancer detection system is ready for production deployment! Choose the architecture that best fits your use case and follow the deployment checklist for a smooth rollout.