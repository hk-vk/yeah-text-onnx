from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime
import numpy as np
import torch
from transformers import AutoTokenizer
import os
import gc
import sys
import importlib.util

# Server configuration
HOST = "127.0.0.1"
PORT = 8080

app = FastAPI(
    title="Malayalam Text Classification API",
    description="API for Malayalam text classification using ONNX model",
    version="1.0.0"
)

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predicted_class: int
    confidence: float

# Initialize model and tokenizer
MODEL_DIR = "model"
MODEL_FILENAME = "malayalam_model1.onnx"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Configure ONNX Runtime session options for memory efficiency
def create_model_session(model_path):
    # Configure session options
    sess_options = onnxruntime.SessionOptions()
    # Set graph optimization level
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    # Limit memory usage
    sess_options.enable_mem_pattern = False
    sess_options.enable_cpu_mem_arena = False
    # Set intra/inter op num threads
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    
    # Create session with CPU provider and optimized settings
    providers = [
        ('CPUExecutionProvider', {
            'cpu_memory_limit': 2 * 1024 * 1024 * 1024,  # 2GB memory limit
        })
    ]
    
    # Force garbage collection before loading model
    gc.collect()
    
    return onnxruntime.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=providers
    )

def ensure_model_exists():
    """Check if model exists, if not, try to download it"""
    if os.path.exists(MODEL_PATH):
        return True
    
    print(f"Model file not found: {MODEL_PATH}")
    print("Attempting to download the model...")
    
    # Try to import the download_model module
    try:
        # Check if download_model.py exists
        if os.path.exists("download_model.py"):
            # Import the module dynamically
            spec = importlib.util.spec_from_file_location("download_model", "download_model.py")
            download_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(download_module)
            
            # Call the check_and_download_model function
            if download_module.check_and_download_model():
                print("Model downloaded successfully!")
                return True
            else:
                print("Failed to download the model.")
                return False
        else:
            print("download_model.py not found. Cannot download the model automatically.")
            return False
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False

@app.on_event("startup")
async def load_model():
    global ort_session, tokenizer
    
    try:
        # Ensure model exists
        if not ensure_model_exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH} and could not be downloaded")
        
        print("Loading model...")
        # Force garbage collection before loading
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Load model with optimized settings
        ort_session = create_model_session(MODEL_PATH)
        print("Model loaded successfully!")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        print("Tokenizer loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    try:
        # Tokenize with truncation to limit memory usage
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Clear memory before inference
        gc.collect()
        
        # Prepare input for ONNX runtime
        ort_inputs = {
            'input_ids': inputs['input_ids'].cpu().numpy(),
            'attention_mask': inputs['attention_mask'].cpu().numpy(),
            'token_type_ids': inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).cpu().numpy()
        }
        
        # Run inference
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # Process the output
        logits = ort_outputs[0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        predicted_class = int(np.argmax(probabilities, axis=1)[0])
        predicted_probability = float(probabilities[0, predicted_class])
        
        # Clear memory after inference
        gc.collect()
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=predicted_probability
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Malayalam Text Classification API is running",
        "model_status": "loaded" if 'ort_session' in globals() else "not loaded",
        "model_path": MODEL_PATH
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting server on http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, reload=True) 