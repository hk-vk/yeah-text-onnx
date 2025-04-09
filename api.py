from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import gc
import numpy as np
import onnxruntime
import torch
from transformers import AutoTokenizer
from typing import Optional

# --- Model Configuration ---
MODEL_DIR = "model"
MODEL_FILENAME = "malayalam_model1.onnx"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Initialize FastAPI app
app = FastAPI(
    title="Malayalam Text Classification API",
    description="API for classifying Malayalam text using ONNX model",
    version="1.0.0"
)

# Request model
class TextRequest(BaseModel):
    text: str

# Response model
class ClassificationResponse(BaseModel):
    predicted_class: int
    confidence: float
    memory_usage: float

# Global variables for model and tokenizer
ort_session = None
tokenizer = None

def load_model_and_tokenizer():
    global ort_session, tokenizer
    
    if ort_session is None or tokenizer is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=500, detail="Model file not found")
        
        try:
            # Load ONNX model
            sess_options = onnxruntime.SessionOptions()
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
            sess_options.enable_mem_pattern = False
            sess_options.enable_cpu_mem_arena = False
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1

            providers = [('CPUExecutionProvider', {})]
            
            gc.collect()
            ort_session = onnxruntime.InferenceSession(
                MODEL_PATH,
                sess_options=sess_options,
                providers=providers
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.on_event("startup")
async def startup_event():
    load_model_and_tokenizer()

@app.post("/classify", response_model=ClassificationResponse)
async def classify_text(request: TextRequest):
    global ort_session, tokenizer
    
    if ort_session is None or tokenizer is None:
        load_model_and_tokenizer()
    
    try:
        # Tokenize input
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Prepare input for ONNX runtime
        ort_inputs = {
            'input_ids': inputs['input_ids'].cpu().numpy(),
            'attention_mask': inputs['attention_mask'].cpu().numpy(),
            'token_type_ids': inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).cpu().numpy()
        }
        del inputs
        gc.collect()

        # Run inference
        ort_outputs = ort_session.run(None, ort_inputs)
        del ort_inputs
        gc.collect()

        # Process output
        logits = ort_outputs[0]
        probabilities = np.exp(logits.astype(np.float32)) / np.sum(np.exp(logits.astype(np.float32)), axis=1, keepdims=True)
        predicted_class = int(np.argmax(probabilities, axis=1)[0])
        predicted_probability = float(probabilities[0, predicted_class])
        del ort_outputs, logits, probabilities
        gc.collect()
        
        # Get memory usage
        import psutil
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        return ClassificationResponse(
            predicted_class=predicted_class,
            confidence=predicted_probability,
            memory_usage=memory_usage
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 