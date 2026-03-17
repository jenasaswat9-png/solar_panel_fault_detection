"""
FastAPI REST API for Solar Panel Fault Detection.

This API provides endpoints for predicting solar panel faults from images.

Usage:
    # Start the API server
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
    
    # Or run directly
    python api/app.py

Endpoints:
    POST /predict - Upload image and get prediction
    GET /health - Health check
    GET /info - API information
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import io
import base64
import json
import time

import numpy as np
from PIL import Image
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, get_device
from src.predict import Predictor


# Load configuration
config = load_config()

# Initialize FastAPI app
app = FastAPI(
    title="Solar Panel Fault Detection API",
    description="API for detecting faults in solar panel images using computer vision",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor (lazy loading)
predictor = None


def get_predictor():
    """Get or initialize the predictor."""
    global predictor
    if predictor is None:
        # Determine model type and weights path
        model_type = os.environ.get('MODEL_TYPE', 'resnet50')
        
        if model_type == 'resnet50':
            weights_path = os.environ.get('MODEL_WEIGHTS', 'models/checkpoints/best_model.pth')
        else:
            weights_path = os.environ.get('MODEL_WEIGHTS', 'models/yolo_training/weights/best.pt')
        
        print(f"Initializing predictor with {model_type} model...")
        predictor = Predictor(config, model_type=model_type, weights_path=weights_path)
        print("Predictor initialized successfully!")
    
    return predictor


# Pydantic models for request/response
class PredictionResponse(BaseModel):
    fault_type: str
    confidence: float
    inference_time_ms: float
    class_index: Optional[int] = None
    all_probabilities: Optional[Dict[str, float]] = None
    detections: Optional[List[Dict]] = None
    num_detections: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    device: str


class InfoResponse(BaseModel):
    name: str
    version: str
    description: str
    classes: List[str]
    image_size: Dict[str, int]


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Solar Panel Fault Detection API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global predictor
    
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        model_type=os.environ.get('MODEL_TYPE', 'resnet50'),
        device=str(get_device())
    )


@app.get("/info", response_model=InfoResponse)
async def get_info():
    """Get API information."""
    return InfoResponse(
        name="Solar Panel Fault Detection API",
        version="1.0.0",
        description="Computer vision API for detecting solar panel faults",
        classes=config['model']['classes'],
        image_size={
            "width": config['image']['width'],
            "height": config['image']['height']
        }
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict fault type from uploaded image.
    
    Args:
        file: Uploaded image file (jpg, jpeg, or png)
        
    Returns:
        Prediction results including fault type and confidence
    """
    # Validate file type
    allowed_extensions = config['api']['allowed_extensions']
    file_extension = file.filename.split('.')[-1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {allowed_extensions}"
        )
    
    # Check file size
    max_size = config['api']['max_file_size']
    contents = await file.read()
    
    if len(contents) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {max_size / 1024 / 1024:.1f} MB"
        )
    
    try:
        # Load image
        image = Image.open(io.BytesIO(contents))
        
        # Convert to OpenCV format (BGR)
        image_array = np.array(image)
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Get predictor
        pred = get_predictor()
        
        # Save temporary file for prediction
        temp_path = f"/tmp/temp_prediction.{file_extension}"
        cv2.imwrite(temp_path, image_array)
        
        # Run prediction
        result = pred.predict_single(temp_path)
        
        # Clean up temp file
        os.remove(temp_path)
        
        # Build response
        response = PredictionResponse(
            fault_type=result['fault_type'],
            confidence=result['confidence'],
            inference_time_ms=result['inference_time'] * 1000,
            class_index=result.get('class_index'),
            all_probabilities=result.get('all_probabilities'),
            detections=result.get('detections'),
            num_detections=result.get('num_detections')
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict fault types for multiple images.
    
    Args:
        files: List of uploaded image files
        
    Returns:
        List of prediction results
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images allowed per batch"
        )
    
    results = []
    
    for file in files:
        try:
            # Validate file type
            file_extension = file.filename.split('.')[-1].lower()
            if file_extension not in config['api']['allowed_extensions']:
                results.append({
                    "filename": file.filename,
                    "error": f"Invalid file type: {file_extension}"
                })
                continue
            
            # Read and process image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            image_array = np.array(image)
            
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Get predictor
            pred = get_predictor()
            
            # Save temporary file
            temp_path = f"/tmp/temp_{file.filename}"
            cv2.imwrite(temp_path, image_array)
            
            # Predict
            result = pred.predict_single(temp_path)
            result['filename'] = file.filename
            
            # Clean up
            os.remove(temp_path)
            
            results.append(result)
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"predictions": results}


@app.post("/predict_visualize")
async def predict_visualize(file: UploadFile = File(...)):
    """
    Predict fault type and return visualization with bounding boxes.
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON with prediction and base64-encoded visualization image
    """
    # Validate file type
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in config['api']['allowed_extensions']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {config['api']['allowed_extensions']}"
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Get predictor
        pred = get_predictor()
        
        # Save temporary file
        temp_path = f"/tmp/temp_viz.{file_extension}"
        viz_path = f"/tmp/temp_viz_output.{file_extension}"
        cv2.imwrite(temp_path, image_array)
        
        # Predict and visualize
        result = pred.predict_single(temp_path)
        pred.visualize_prediction(temp_path, viz_path)
        
        # Read visualization
        with open(viz_path, "rb") as f:
            viz_bytes = f.read()
        
        # Encode to base64
        viz_base64 = base64.b64encode(viz_bytes).decode('utf-8')
        
        # Clean up
        os.remove(temp_path)
        os.remove(viz_path)
        
        return {
            "prediction": {
                "fault_type": result['fault_type'],
                "confidence": result['confidence'],
                "inference_time_ms": result['inference_time'] * 1000
            },
            "visualization": f"data:image/{file_extension};base64,{viz_base64}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize predictor on startup."""
    print("="*70)
    print("SOLAR PANEL FAULT DETECTION API")
    print("="*70)
    print(f"Model Type: {os.environ.get('MODEL_TYPE', 'resnet50')}")
    print(f"Device: {get_device()}")
    print(f"Classes: {config['model']['classes']}")
    print("="*70)
    print("\nAPI Documentation: http://localhost:8000/docs")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run the API server
    host = config['api']['host']
    port = config['api']['port']
    
    uvicorn.run(app, host=host, port=port)
