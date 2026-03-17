"""
Prediction Pipeline for Solar Panel Fault Detection.

This script provides inference functionality for detecting faults in
solar panel images. It supports both classification and object detection
models.

Usage:
    # Predict on single image
    python src/predict.py --image path/to/image.jpg --model resnet50 --weights models/best.pth
    
    # Predict on directory
    python src/predict.py --input-dir path/to/images/ --output-dir results/predictions/
    
    # Save visualization with bounding boxes
    python src/predict.py --image image.jpg --save-viz --output prediction.jpg
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import time

import numpy as np
import torch
import cv2
from PIL import Image

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, get_device, set_seed, draw_bounding_boxes
from src.preprocess import ImagePreprocessor
from src.train import ResNet50Classifier


class Predictor:
    """Predictor class for solar panel fault detection."""
    
    def __init__(self, config: Dict, model_type: str = 'resnet50',
                 weights_path: str = None, device: torch.device = None):
        """
        Initialize predictor.
        
        Args:
            config: Configuration dictionary
            model_type: Type of model ('resnet50' or 'yolo')
            weights_path: Path to model weights
            device: Device to run inference on
        """
        self.config = config
        self.model_type = model_type
        self.device = device or get_device()
        
        # Class names
        self.class_names = config['model']['classes']
        self.num_classes = len(self.class_names)
        
        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor(config)
        
        # Load model
        self._load_model(weights_path)
    
    def _load_model(self, weights_path: str) -> None:
        """Load the trained model."""
        if self.model_type == 'resnet50':
            self.model = ResNet50Classifier(
                num_classes=self.num_classes,
                pretrained=False
            )
            
            # Load weights
            if weights_path and os.path.exists(weights_path):
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded ResNet50 weights from {weights_path}")
            else:
                print("Warning: No weights loaded - using random initialization")
            
            self.model.to(self.device)
            self.model.eval()
            
        elif self.model_type == 'yolo':
            from ultralytics import YOLO
            
            if weights_path and os.path.exists(weights_path):
                self.model = YOLO(weights_path)
            else:
                # Load default YOLO model
                model_name = self.config['model']['name']
                self.model = YOLO(f'{model_name}.pt')
                print(f"Loaded default YOLO model: {model_name}")
            
            print(f"Loaded YOLO model from {weights_path}")
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def predict_single(self, image_path: str, 
                      return_confidence: bool = True) -> Dict:
        """
        Predict on a single image.
        
        Args:
            image_path: Path to input image
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary containing prediction results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Run prediction
        start_time = time.time()
        
        if self.model_type == 'resnet50':
            result = self._predict_resnet50(image, return_confidence)
        elif self.model_type == 'yolo':
            result = self._predict_yolo(image, return_confidence)
        
        inference_time = time.time() - start_time
        result['inference_time'] = inference_time
        result['image_path'] = image_path
        
        return result
    
    def _predict_resnet50(self, image: np.ndarray, 
                         return_confidence: bool = True) -> Dict:
        """
        Predict using ResNet50 model.
        
        Args:
            image: Input image (BGR)
            return_confidence: Whether to return confidence scores
            
        Returns:
            Prediction results dictionary
        """
        # Preprocess
        input_tensor = self.preprocessor.preprocess_for_inference(image)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        # Get results
        pred_class_idx = predicted.item()
        pred_class_name = self.class_names[pred_class_idx]
        pred_confidence = confidence.item()
        
        result = {
            'fault_type': pred_class_name,
            'confidence': pred_confidence,
            'class_index': pred_class_idx
        }
        
        if return_confidence:
            # Return all class probabilities
            all_probs = probabilities[0].cpu().numpy()
            result['all_probabilities'] = {
                class_name: float(prob)
                for class_name, prob in zip(self.class_names, all_probs)
            }
        
        return result
    
    def _predict_yolo(self, image: np.ndarray,
                     return_confidence: bool = True) -> Dict:
        """
        Predict using YOLO model.
        
        Args:
            image: Input image (BGR)
            return_confidence: Whether to return confidence scores
            
        Returns:
            Prediction results dictionary
        """
        # YOLO expects RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model(
            image_rgb,
            verbose=False,
            conf=self.config['evaluation']['conf_threshold'],
            iou=self.config['evaluation']['iou_threshold']
        )[0]
        
        # Parse results
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_idx in zip(boxes, confidences, classes):
                detection = {
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'class': self.class_names[cls_idx] if cls_idx < len(self.class_names) else 'unknown',
                    'class_index': int(cls_idx)
                }
                detections.append(detection)
        
        # Get primary prediction (highest confidence)
        if detections:
            best_detection = max(detections, key=lambda x: x['confidence'])
            result = {
                'fault_type': best_detection['class'],
                'confidence': best_detection['confidence'],
                'detections': detections,
                'num_detections': len(detections)
            }
        else:
            result = {
                'fault_type': 'normal',
                'confidence': 1.0,
                'detections': [],
                'num_detections': 0
            }
        
        return result
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Predict on multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'error': str(e),
                    'image_path': image_path
                })
        
        return results
    
    def visualize_prediction(self, image_path: str, 
                            output_path: str = None,
                            show_confidence: bool = True) -> np.ndarray:
        """
        Visualize prediction on image with bounding boxes.
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization
            show_confidence: Whether to show confidence scores
            
        Returns:
            Visualization image
        """
        # Load image
        image = cv2.imread(image_path)
        
        # Get prediction
        result = self.predict_single(image_path)
        
        # Create visualization
        viz_image = image.copy()
        
        if self.model_type == 'yolo' and 'detections' in result:
            # Draw YOLO detections
            for detection in result['detections']:
                bbox = detection['bbox']
                label = detection['class']
                conf = detection['confidence']
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # Color based on class
                color_map = {
                    'normal': (0, 255, 0),
                    'cracked': (0, 0, 255),
                    'hotspot': (255, 0, 0),
                    'dust': (255, 255, 0)
                }
                color = color_map.get(label, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                if show_confidence:
                    text = f"{label}: {conf:.2f}"
                else:
                    text = label
                
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                cv2.rectangle(viz_image, (x1, y1 - text_height - 10),
                            (x1 + text_width, y1), color, -1)
                cv2.putText(viz_image, text, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        else:
            # For classification model, show prediction text
            fault_type = result.get('fault_type', 'unknown')
            confidence = result.get('confidence', 0.0)
            
            text = f"Prediction: {fault_type} ({confidence:.2f})"
            cv2.putText(viz_image, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save if output path specified
        if output_path:
            cv2.imwrite(output_path, viz_image)
            print(f"Visualization saved to {output_path}")
        
        return viz_image


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Predict Solar Panel Faults')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to input image')
    parser.add_argument('--input-dir', type=str, default=None,
                       help='Directory containing input images')
    parser.add_argument('--output-dir', type=str, default='results/predictions',
                       help='Directory to save predictions')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'yolo'],
                       help='Model type')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to model weights')
    parser.add_argument('--save-viz', action='store_true',
                       help='Save visualization with bounding boxes')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for single image prediction')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--save-json', action='store_true',
                       help='Save predictions to JSON file')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Default weights path
    if args.weights is None:
        if args.model == 'resnet50':
            args.weights = 'models/checkpoints/best_model.pth'
        else:
            args.weights = 'models/yolo_training/weights/best.pt'
    
    # Create predictor
    predictor = Predictor(config, model_type=args.model, weights_path=args.weights)
    
    # Single image prediction
    if args.image:
        print(f"\nPredicting on: {args.image}")
        print("-" * 50)
        
        result = predictor.predict_single(args.image)
        
        print(f"Fault Type: {result['fault_type']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Inference Time: {result['inference_time']*1000:.2f} ms")
        
        if 'all_probabilities' in result:
            print("\nAll Probabilities:")
            for class_name, prob in result['all_probabilities'].items():
                print(f"  {class_name}: {prob:.4f}")
        
        if 'detections' in result:
            print(f"\nDetections: {result['num_detections']}")
            for i, det in enumerate(result['detections']):
                print(f"  {i+1}. {det['class']} ({det['confidence']:.4f})")
        
        # Save visualization
        if args.save_viz or args.output:
            output_path = args.output or 'prediction_result.jpg'
            predictor.visualize_prediction(args.image, output_path)
    
    # Batch prediction
    elif args.input_dir:
        if not os.path.exists(args.input_dir):
            print(f"Input directory not found: {args.input_dir}")
            return
        
        # Get all images
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(Path(args.input_dir).glob(ext))
        
        print(f"\nFound {len(image_paths)} images in {args.input_dir}")
        print("-" * 50)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Run predictions
        results = []
        for img_path in image_paths:
            print(f"Processing: {img_path.name}...", end=' ')
            
            result = predictor.predict_single(str(img_path))
            results.append(result)
            
            print(f"{result['fault_type']} ({result['confidence']:.2f})")
            
            # Save visualization
            if args.save_viz:
                output_path = os.path.join(
                    args.output_dir,
                    f"{img_path.stem}_prediction{img_path.suffix}"
                )
                predictor.visualize_prediction(str(img_path), output_path)
        
        # Save results to JSON
        if args.save_json:
            json_path = os.path.join(args.output_dir, 'predictions.json')
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nPredictions saved to {json_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("PREDICTION SUMMARY")
        print("="*50)
        
        fault_counts = {}
        for result in results:
            fault_type = result['fault_type']
            fault_counts[fault_type] = fault_counts.get(fault_type, 0) + 1
        
        for fault_type, count in sorted(fault_counts.items()):
            percentage = (count / len(results)) * 100
            print(f"{fault_type:15s}: {count:3d} ({percentage:5.1f}%)")
        
        print("-"*50)
        print(f"Total: {len(results)} images")
        print("="*50)
    
    else:
        print("Please specify --image or --input-dir")


if __name__ == "__main__":
    main()
