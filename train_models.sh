#!/bin/bash
# Script to train both models sequentially

echo "Starting Oral Cancer Detection Model Training"
echo "==========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate || venv\Scripts\activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating output directories..."
mkdir -p outputs/{models,logs,figures,results}

# Train RegNetY320
echo ""
echo "Training RegNetY320 Model..."
echo "---------------------------"
python src/training/train.py --model regnet --data_dir data --epochs 100 --device cuda --save_dir outputs

# Train VGG16
echo ""
echo "Training VGG16 Model..."
echo "----------------------"
python src/training/train.py --model vgg16 --data_dir data --epochs 100 --device cuda --save_dir outputs

# Compare models
echo ""
echo "Comparing Models..."
echo "------------------"
python src/evaluation/compare_models.py --data_dir data --model_dir outputs/models --save_dir outputs

echo ""
echo "Training Complete!"
echo "=================="
echo "Check outputs/evaluation_report.txt for detailed results"
echo "View training logs with: tensorboard --logdir outputs/logs"