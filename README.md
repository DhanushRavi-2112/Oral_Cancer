# Oral Cancer Detection using RegNetY320 vs VGG16

This project implements a binary classification system for oral cancer detection using deep learning. It compares two state-of-the-art CNN architectures: RegNetY320 (efficient modern architecture) and VGG16 (classical architecture) to determine the optimal model for clinical deployment.

## Project Structure

```
Oral_cancer/
├── data/                      # Dataset directory
│   ├── train/
│   │   ├── healthy/          # Healthy tissue images
│   │   └── cancer/           # Cancer tissue images
│   ├── val/
│   │   ├── healthy/
│   │   └── cancer/
│   └── test/
│       ├── healthy/
│       └── cancer/
├── src/                       # Source code
│   ├── data/
│   │   └── preprocessing.py   # Data loading and augmentation
│   ├── models/
│   │   └── architectures.py   # Model definitions
│   ├── training/
│   │   └── train.py          # Training script
│   └── evaluation/
│       └── compare_models.py  # Model comparison framework
├── outputs/                   # Training outputs
│   ├── models/               # Saved model checkpoints
│   ├── logs/                 # TensorBoard logs
│   ├── figures/              # Visualizations
│   └── results/              # Evaluation results
├── predict.py                # Inference script
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 32GB RAM (minimum)
- 500GB storage for datasets and models

## Installation

1. Clone the repository:
```bash
cd Oral_cancer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Organize your dataset in the following structure:
```
data/
├── train/
│   ├── healthy/  (place healthy tissue images here)
│   └── cancer/   (place cancer tissue images here)
├── val/
│   ├── healthy/
│   └── cancer/
└── test/
    ├── healthy/
    └── cancer/
```

2. Ensure images are:
   - High resolution (minimum 512x512)
   - In JPEG or PNG format
   - Properly labeled by medical professionals

3. Recommended split:
   - Training: 70% (minimum 3,500 images)
   - Validation: 15% (minimum 750 images)
   - Test: 15% (minimum 750 images)

## Training Models

### Train RegNetY320

```bash
python src/training/train.py --model regnet --data_dir data --epochs 100 --device cuda
```

### Train VGG16

```bash
python src/training/train.py --model vgg16 --data_dir data --epochs 100 --device cuda
```

### Training Options

- `--model`: Model type (`regnet` or `vgg16`)
- `--data_dir`: Path to data directory (default: `data`)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (uses model defaults if not specified)
- `--lr`: Learning rate (uses model defaults if not specified)
- `--device`: Device to train on (`cuda` or `cpu`)
- `--save_dir`: Output directory (default: `outputs`)

## Model Evaluation and Comparison

After training both models, compare their performance:

```bash
python src/evaluation/compare_models.py --data_dir data --model_dir outputs/models --save_dir outputs
```

This will generate:
- Performance metrics comparison
- ROC curves
- Confusion matrices
- Statistical significance tests
- Comprehensive evaluation report

## Inference

### Single Image Prediction

```bash
python predict.py path/to/image.jpg --model outputs/models/RegNetY-320MF/best_model.pth --model_type regnet
```

### Batch Prediction (Directory)

```bash
python predict.py path/to/image/directory --model outputs/models/VGG16/best_model.pth --model_type vgg16 --output results.json
```

### Inference Options

- `--model`: Path to trained model checkpoint
- `--model_type`: Model architecture (`regnet` or `vgg16`)
- `--threshold`: Confidence threshold (default: 0.5)
- `--visualize`: Generate visualization of predictions
- `--output`: Save results to JSON file
- `--batch_size`: Batch size for processing multiple images

## Model Performance

### Target Metrics
- **Sensitivity (Recall)**: > 95% (critical for cancer detection)
- **Specificity**: > 90%
- **F1-Score**: > 92%
- **AUC-ROC**: > 0.95

### Model Comparison

| Model | Parameters | Inference Time | Expected Performance |
|-------|------------|----------------|---------------------|
| RegNetY320 | ~3.2M | ~10ms | High efficiency, suitable for edge deployment |
| VGG16 | ~138M | ~25ms | Proven accuracy, higher resource requirements |

## Key Features

1. **Data Augmentation**:
   - Random rotations, flips, brightness/contrast adjustments
   - Designed to improve model generalization

2. **Progressive Training**:
   - Transfer learning from ImageNet
   - Progressive unfreezing for VGG16
   - Early stopping to prevent overfitting

3. **Comprehensive Evaluation**:
   - Multiple metrics (accuracy, sensitivity, specificity, F1, AUC)
   - Statistical significance testing
   - Clinical deployment recommendations

4. **Production Ready**:
   - Efficient inference pipeline
   - Batch processing support
   - Visualization tools

## Monitoring Training

View training progress using TensorBoard:

```bash
tensorboard --logdir outputs/logs
```

## Results Interpretation

After evaluation, check `outputs/evaluation_report.txt` for:
- Detailed performance metrics
- Model recommendations based on:
  - Best sensitivity (cancer detection rate)
  - Best specificity (avoiding false positives)
  - Best overall balance (F1-Score)
  - Fastest inference time

## Clinical Deployment Considerations

1. **Model Selection**:
   - For screening: Prioritize sensitivity (RegNetY320 if it meets >95%)
   - For diagnosis: Balance sensitivity and specificity
   - For mobile/edge: RegNetY320 (smaller, faster)

2. **Confidence Thresholds**:
   - Adjust threshold based on clinical requirements
   - Lower threshold = higher sensitivity, lower specificity
   - Higher threshold = lower sensitivity, higher specificity

3. **Human Review**:
   - Always require expert review for positive predictions
   - Use confidence scores to prioritize cases

## Troubleshooting

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Use gradient accumulation
   - Switch to CPU (slower)

2. **Poor Performance**:
   - Check data quality and labeling
   - Ensure balanced dataset
   - Increase training epochs
   - Adjust learning rate

3. **Slow Training**:
   - Enable mixed precision training
   - Use multiple GPUs
   - Optimize data loading workers

## Future Enhancements

1. Implement multi-class classification (benign/pre-malignant/malignant)
2. Add explainability (Grad-CAM visualizations)
3. Integrate with PACS systems
4. Mobile app deployment
5. Real-time video stream processing

## Citation

If you use this project in your research, please cite:

```
@software{oral_cancer_detection,
  title={Binary Classification of Oral Cancer using RegNetY320 vs VGG16},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/oral-cancer-detection}
}
```

## License

This project is for educational and research purposes. For clinical use, ensure proper regulatory approval and validation.

## Support

For questions or issues:
1. Check existing issues in the repository
2. Review the documentation
3. Contact the maintainers

Remember: This system is designed to assist medical professionals, not replace them. Always consult qualified healthcare providers for medical decisions.
