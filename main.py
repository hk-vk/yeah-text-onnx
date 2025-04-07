import os
import gc
import sys
import importlib.util
import psutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime
import numpy as np
import torch
from transformers import AutoTokenizer

def log_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")  # Default to 0.0.0.0 for production
PORT = int(os.getenv("PORT", 8080))  # Use Railway's PORT

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

@app.on_event("startup")
async def load_model():
    global ort_session, tokenizer
    
    try:
        # Log initial memory usage
        log_memory_usage()
        
        # Check if the model file exists (should be present via Git LFS)
        if not os.path.exists(MODEL_PATH):
            error_msg = f"Model file not found at {MODEL_PATH}. Ensure Git LFS is configured correctly and the file was checked out."
            print(error_msg)
            raise FileNotFoundError(error_msg)
        
        print("Loading model...")
        # Force garbage collection before loading
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Load model with optimized settings
        ort_session = create_model_session(MODEL_PATH)
        print("Model loaded successfully!")
        
        # Load tokenizer - assumes tokenizer files are also in MODEL_DIR and tracked by Git
        # Ensure tokenizer files (config.json, etc.) are NOT ignored by .gitignore
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        print("Tokenizer loaded successfully!")
        
        # Log final memory usage
        gc.collect()
        log_memory_usage()
        
    except FileNotFoundError as e:
        # Specific handling for model file not found
        print(f"Startup Error: {str(e)}")
        # Don't raise HTTPException here, let the server fail to start if model is missing
        # A running server without a model is not useful.
        sys.exit(f"Critical error: Model file missing at startup: {MODEL_PATH}")
    except Exception as e:
        # Catch other potential errors during startup (e.g., ONNX loading, tokenizer loading)
        print(f"Error loading model/tokenizer: {str(e)}")
        # Optional: include traceback
        # import traceback
        # print(traceback.format_exc())
        # Exit prevents the server starting in a broken state
        sys.exit(f"Critical startup error: {str(e)}")

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
        
        # Explicitly delete PyTorch tensors to free memory sooner
        del inputs
        gc.collect() # Add another GC call after deletion
        
        # Run inference
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # Process the output
        logits = ort_outputs[0]
        # Convert logits directly to float for calculations if possible
        # This avoids keeping the potentially large numpy array around longer than needed
        probabilities = np.exp(logits.astype(np.float32)) / np.sum(np.exp(logits.astype(np.float32)), axis=1, keepdims=True)
        predicted_class = int(np.argmax(probabilities, axis=1)[0])
        predicted_probability = float(probabilities[0, predicted_class])
        
        # Explicitly delete large intermediate NumPy arrays
        del ort_inputs
        del ort_outputs
        del logits
        del probabilities
        gc.collect() # Add another GC call after processing
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=predicted_probability
        )
    except Exception as e:
        # Log the exception for debugging
        print(f"Error during prediction: {str(e)}")
        # Optional: include traceback
        # import traceback
        # print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

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