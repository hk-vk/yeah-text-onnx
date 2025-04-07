import streamlit as st
import os
import gc
import psutil
import numpy as np
import onnxruntime
import torch # Still needed for tensor creation, though conversion is internal now
from transformers import AutoTokenizer

# --- Model Configuration ---
MODEL_DIR = "model"
MODEL_FILENAME = "malayalam_model1.onnx"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# --- Helper Functions (from main.py) ---
def log_memory_usage():
    process = psutil.Process(os.getpid())
    st.sidebar.metric("Memory Usage", f"{process.memory_info().rss / 1024 / 1024:.2f} MB")

# --- Cached Model Loading --- 
# Use @st.cache_resource to load model and tokenizer only once
@st.cache_resource
def load_onnx_model_and_tokenizer():
    st.sidebar.write("Loading ONNX model...")
    log_memory_usage()
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Ensure Git LFS checked out the file.")
        st.stop()
        
    try:
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess_options.enable_mem_pattern = False
        sess_options.enable_cpu_mem_arena = False
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1

        providers = [('CPUExecutionProvider', {})] # Simpler provider config for now
        
        gc.collect()
        ort_session = onnxruntime.InferenceSession(
            MODEL_PATH,
            sess_options=sess_options,
            providers=providers
        )
        st.sidebar.success("ONNX model loaded.")
        log_memory_usage()
    except Exception as e:
        st.error(f"Error loading ONNX model: {e}")
        st.stop()

    st.sidebar.write("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        st.sidebar.success("Tokenizer loaded.")
        log_memory_usage()
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        st.stop()
        
    return ort_session, tokenizer

# --- Streamlit App UI --- 
st.title("Malayalam Text Classification")

# Load model and tokenizer using the cached function
ort_session, tokenizer = load_onnx_model_and_tokenizer()

text_input = st.text_area("Enter Malayalam Text:", height=150)

if st.button("Classify Text"):
    if text_input and ort_session and tokenizer:
        try:
            st.write("Tokenizing...")
            # Tokenize 
            inputs = tokenizer(
                text_input,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            st.write("Preparing inputs for ONNX...")
            # Prepare input for ONNX runtime
            ort_inputs = {
                'input_ids': inputs['input_ids'].cpu().numpy(),
                'attention_mask': inputs['attention_mask'].cpu().numpy(),
                'token_type_ids': inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).cpu().numpy()
            }
            del inputs # Clean up tensor
            gc.collect()

            st.write("Running inference...")
            # Run inference
            ort_outputs = ort_session.run(None, ort_inputs)
            del ort_inputs # Clean up input dict
            gc.collect()

            st.write("Processing results...")
            # Process the output
            logits = ort_outputs[0]
            probabilities = np.exp(logits.astype(np.float32)) / np.sum(np.exp(logits.astype(np.float32)), axis=1, keepdims=True)
            predicted_class = int(np.argmax(probabilities, axis=1)[0])
            predicted_probability = float(probabilities[0, predicted_class])
            del ort_outputs, logits, probabilities # Clean up output arrays
            gc.collect()
            
            st.success("Classification successful!")
            st.metric(label="Predicted Class", value=predicted_class)
            st.metric(label="Confidence", value=f"{predicted_probability:.2%}")
            log_memory_usage() # Log memory after prediction
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            # import traceback
            # st.error(traceback.format_exc())
    elif not text_input:
        st.warning("Please enter some text to classify.")
    else:
        st.error("Model or Tokenizer not loaded correctly. Check sidebar.")

# Add a button to clear cache for debugging model loading
st.sidebar.button("Clear Model Cache", on_click=load_onnx_model_and_tokenizer.clear) 