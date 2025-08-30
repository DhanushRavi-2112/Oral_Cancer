import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.preprocessing import create_data_loaders
from src.models.architectures import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


class Trainer:
    """Trainer class for oral cancer detection models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        config: Dict,
        device: str = 'cuda',
        save_dir: str = 'outputs'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.save_dir = save_dir
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup loss function with class weights
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([config.get('pos_weight', 1.0)]).to(device)
        )
        
        # Setup scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 15)
        )
        
        # Setup tensorboard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(
            f"{save_dir}/logs/{config['model_name']}_{timestamp}"
        )
        
        # Tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.best_val_auc = 0.0
        
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on config."""
        opt_name = self.config.get('optimizer', 'Adam')
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.0001)
        
        if opt_name == 'Adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_name == 'SGD':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def train_epoch(self, epoch: int) -> Tuple[float, Dict]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.float().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images).squeeze()
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        # Calculate epoch metrics
        avg_loss = running_loss / len(self.train_loader)
        metrics = self._calculate_metrics(all_labels, all_preds)
        
        return avg_loss, metrics
    
    def validate(self, epoch: int) -> Tuple[float, Dict]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch}")
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.float().to(self.device)
                
                outputs = self.model(images).squeeze()
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = running_loss / len(self.val_loader)
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        
        return avg_loss, metrics
    
    def _calculate_metrics(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        probs: Optional[np.ndarray] = None
    ) -> Dict:
        """Calculate classification metrics."""
        labels = np.array(labels)
        preds = np.array(preds)
        
        # Binary predictions
        binary_preds = (preds > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(labels, binary_preds),
            'precision': precision_score(labels, binary_preds, zero_division=0),
            'recall': recall_score(labels, binary_preds, zero_division=0),
            'f1': f1_score(labels, binary_preds, zero_division=0),
        }
        
        # Calculate AUC if probabilities are provided
        if probs is not None:
            try:
                metrics['auc'] = roc_auc_score(labels, probs)
            except:
                metrics['auc'] = 0.0
        
        # Calculate specificity
        tn = ((labels == 0) & (binary_preds == 0)).sum()
        fp = ((labels == 0) & (binary_preds == 1)).sum()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['specificity'] = specificity
        
        return metrics
    
    def train(self, epochs: int, save_best: bool = True):
        """Train the model for specified epochs."""
        logger.info(f"Starting training for {epochs} epochs")
        
        # Progressive unfreezing for VGG16
        unfreeze_schedule = None
        if self.config['model_name'] == 'VGG16':
            unfreeze_schedule = {20: 6, 50: -1}  # Unfreeze 6 layers at epoch 20, all at epoch 50
        
        for epoch in range(1, epochs + 1):
            # Check for progressive unfreezing
            if unfreeze_schedule and epoch in unfreeze_schedule:
                self.model.unfreeze_layers(unfreeze_schedule[epoch])
                logger.info(f"Unfreezing layers at epoch {epoch}")
            
            # Training
            train_loss, train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metrics)
            
            # Validation
            val_loss, val_metrics = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Log metrics
            self._log_metrics(epoch, train_loss, train_metrics, val_loss, val_metrics)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if save_best and val_metrics.get('auc', 0) > self.best_val_auc:
                self.best_val_auc = val_metrics.get('auc', 0)
                self.save_checkpoint(epoch, is_best=True)
            
            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                # Ensure best model is saved before stopping
                if not save_best or self.best_val_auc == 0:
                    self.save_checkpoint(epoch, is_best=True)
                break
        
        self.writer.close()
        logger.info("Training completed")
    
    def _log_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_metrics: Dict,
        val_loss: float,
        val_metrics: Dict
    ):
        """Log metrics to console and tensorboard."""
        # Console logging
        logger.info(f"\nEpoch {epoch}")
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Train Metrics: {train_metrics}")
        logger.info(f"Val Metrics: {val_metrics}")
        
        # Tensorboard logging
        self.writer.add_scalars('Loss', {
            'train': train_loss,
            'val': val_loss
        }, epoch)
        
        for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity']:
            if metric_name in train_metrics:
                self.writer.add_scalars(f'Metrics/{metric_name}', {
                    'train': train_metrics[metric_name],
                    'val': val_metrics.get(metric_name, 0)
                }, epoch)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'config': self.config
        }
        
        # Save checkpoint
        checkpoint_dir = os.path.join(self.save_dir, 'models', self.config['model_name'])
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if is_best:
            path = os.path.join(checkpoint_dir, 'best_model.pth')
            logger.info(f"Saving best model with AUC: {self.best_val_auc:.4f}")
        else:
            path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, path)
        logger.info(f"Model checkpoint saved to: {path}")
    
    def test(self) -> Dict:
        """Test the model on test set."""
        logger.info("Testing model on test set...")
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Testing"):
                images = images.to(self.device)
                labels = labels.float()
                
                outputs = self.model(images).squeeze()
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        test_metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        
        logger.info(f"Test Metrics: {test_metrics}")
        
        # Save test results
        results_path = os.path.join(
            self.save_dir, 
            'results', 
            f"{self.config['model_name']}_test_results.json"
        )
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump({
                'metrics': test_metrics,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }, f, indent=4)
        
        return test_metrics


def main():
    parser = argparse.ArgumentParser(description='Train oral cancer detection models')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['regnet', 'vgg16'],
                       help='Model architecture to train')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (uses model default if not specified)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (uses model default if not specified)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='outputs',
                       help='Directory to save outputs')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and args.device == 'cuda':
        logger.warning("CUDA not available, using CPU")
    
    # Create model and get config
    model, config = create_model(args.model, pretrained=True, device=device)
    
    # Override config with command line arguments
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        args.data_dir,
        batch_size=config['batch_size'],
        model_type=args.model
    )
    
    # Add class weight to config for loss function
    config['pos_weight'] = class_weights[1] / class_weights[0]  # cancer/healthy ratio
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        save_dir=args.save_dir
    )
    
    # Train model
    trainer.train(epochs=args.epochs, save_best=True)
    
    # Test model
    test_metrics = trainer.test()
    
    logger.info("Training and testing completed!")


if __name__ == "__main__":
    main()