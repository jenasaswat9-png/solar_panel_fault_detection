"""
Model Evaluation Script for Solar Panel Fault Detection.

This script evaluates trained models and generates comprehensive metrics
including accuracy, precision, recall, F1-score, confusion matrix, and mAP.

Usage:
    # Evaluate ResNet50 model
    python src/evaluate.py --model resnet50 --weights models/checkpoints/best_model.pth
    
    # Evaluate YOLO model
    python src/evaluate.py --model yolo --weights models/yolov8n.pt
    
    # Evaluate on specific split
    python src/evaluate.py --split test
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import (
    load_config, setup_logging, get_device, set_seed,
    plot_confusion_matrix, plot_training_history, calculate_metrics
)
from src.data_loader import create_data_loaders, get_val_transforms, SolarPanelDataset
from src.train import ResNet50Classifier


class Evaluator:
    """Evaluator class for solar panel fault detection models."""
    
    def __init__(self, config: Dict, model_type: str = 'resnet50', 
                 weights_path: str = None):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
            model_type: Type of model ('resnet50' or 'yolo')
            weights_path: Path to model weights
        """
        self.config = config
        self.model_type = model_type
        self.weights_path = weights_path
        self.device = get_device()
        self.logger = setup_logging()
        
        # Class names
        self.class_names = config['model']['classes']
        
        # Load model
        self._load_model()
        
        # Load data
        self._load_data()
    
    def _load_model(self) -> None:
        """Load the trained model."""
        num_classes = len(self.class_names)
        
        if self.model_type == 'resnet50':
            self.model = ResNet50Classifier(
                num_classes=num_classes,
                pretrained=False
            )
            
            # Load weights
            if self.weights_path and os.path.exists(self.weights_path):
                checkpoint = torch.load(self.weights_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"Loaded weights from {self.weights_path}")
            else:
                self.logger.warning("No weights loaded - using random initialization")
            
            self.model.to(self.device)
            self.model.eval()
            
        elif self.model_type == 'yolo':
            from ultralytics import YOLO
            
            if self.weights_path and os.path.exists(self.weights_path):
                self.model = YOLO(self.weights_path)
            else:
                # Load default YOLO model
                model_name = self.config['model']['name']
                self.model = YOLO(f'{model_name}.pt')
            
            self.logger.info(f"Loaded YOLO model from {self.weights_path}")
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _load_data(self, split: str = 'test') -> None:
        """
        Load test data.
        
        Args:
            split: Data split to load ('train', 'val', or 'test')
        """
        data_dir = self.config['dataset']['processed_dir']
        
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        test_dir = os.path.join(data_dir, 'test')
        
        if self.model_type == 'resnet50':
            _, _, self.test_loader, self.class_mapping = create_data_loaders(
                train_dir=train_dir,
                val_dir=val_dir,
                test_dir=test_dir,
                batch_size=self.config['training']['batch_size'],
                num_workers=4,
                config=self.config
            )
        else:
            # For YOLO, just store the test directory
            self.test_dir = test_dir
    
    def evaluate_resnet50(self) -> Dict:
        """
        Evaluate ResNet50 model.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info("="*70)
        self.logger.info("EVALUATING RESNET50 MODEL")
        self.logger.info("="*70)
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Evaluating'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                # Collect predictions
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        
        # Generate visualizations
        self._generate_visualizations(all_labels, all_preds, all_probs)
        
        return metrics
    
    def evaluate_yolo(self) -> Dict:
        """
        Evaluate YOLO model.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info("="*70)
        self.logger.info("EVALUATING YOLO MODEL")
        self.logger.info("="*70)
        
        # Run YOLO validation
        results = self.model.val(
            data=os.path.join(self.config['dataset']['processed_dir'], 'data.yaml'),
            split='test',
            imgsz=self.config['image']['width'],
            batch=self.config['training']['batch_size'],
            device=self.device,
            save_json=True,
            save_hybrid=True,
            conf=self.config['evaluation']['conf_threshold'],
            iou=self.config['evaluation']['iou_threshold']
        )
        
        # Extract metrics
        metrics = {
            'mAP50': results.results_dict.get('metrics/mAP50', 0),
            'mAP50-95': results.results_dict.get('metrics/mAP50-95', 0),
            'precision': results.results_dict.get('metrics/precision', 0),
            'recall': results.results_dict.get('metrics/recall', 0),
            'f1_score': results.results_dict.get('metrics/F1', 0)
        }
        
        self.logger.info(f"YOLO Evaluation Results:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: np.ndarray = None) -> Dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Log metrics
        self.logger.info("\n" + "="*70)
        self.logger.info("EVALUATION METRICS")
        self.logger.info("="*70)
        self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"Precision (macro): {metrics['precision_macro']:.4f}")
        self.logger.info(f"Recall (macro): {metrics['recall_macro']:.4f}")
        self.logger.info(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
        self.logger.info("-"*70)
        
        # Classification report
        self.logger.info("\nClassification Report:")
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4
        )
        self.logger.info("\n" + report)
        
        return metrics
    
    def _generate_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_prob: np.ndarray = None) -> None:
        """
        Generate evaluation visualizations.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
        """
        os.makedirs('results', exist_ok=True)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(
            cm, self.class_names,
            save_path='results/confusion_matrix.png',
            normalize=False
        )
        
        # Normalized Confusion Matrix
        plot_confusion_matrix(
            cm, self.class_names,
            save_path='results/confusion_matrix_normalized.png',
            normalize=True
        )
        
        # Per-class metrics bar plot
        self._plot_per_class_metrics(y_true, y_pred)
        
        # ROC curves (if probabilities available)
        if y_prob is not None:
            self._plot_roc_curves(y_true, y_prob)
    
    def _plot_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot per-class precision, recall, and F1-score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Metrics', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/per_class_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Per-class metrics plot saved to results/per_class_metrics.png")
    
    def _plot_roc_curves(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """
        Plot ROC curves for each class.
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities
        """
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Compute ROC curve and ROC area for each class
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("ROC curves saved to results/roc_curves.png")
    
    def evaluate(self) -> Dict:
        """
        Run evaluation.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model_type == 'resnet50':
            return self.evaluate_resnet50()
        elif self.model_type == 'yolo':
            return self.evaluate_yolo()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Solar Panel Fault Detection Model')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'yolo'],
                       help='Model type to evaluate')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to model weights')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Data split to evaluate on')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    set_seed(args.seed)
    
    # Default weights path if not specified
    if args.weights is None:
        if args.model == 'resnet50':
            args.weights = 'models/checkpoints/best_model.pth'
        else:
            args.weights = 'models/yolov8n.pt'
    
    # Create evaluator
    evaluator = Evaluator(config, model_type=args.model, weights_path=args.weights)
    
    # Evaluate
    metrics = evaluator.evaluate()
    
    # Save metrics
    os.makedirs('results', exist_ok=True)
    metrics_path = 'results/evaluation_metrics.json'
    
    # Convert numpy types to native Python types for JSON serialization
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_serializable[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            metrics_serializable[key] = float(value)
        else:
            metrics_serializable[key] = value
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print(f"\nEvaluation metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
