FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create model directory and download the model
RUN mkdir -p model && \
    echo "Downloading model file..." && \
    gdown --fuzzy "https://drive.google.com/uc?id=1UfWvrS04ssG3WlDeV_uaqetrSZ70b6cW" -O model/malayalam_model1.onnx && \
    echo "Verifying model file..." && \
    if [ ! -f model/malayalam_model1.onnx ]; then \
        echo "ERROR: Model file not found after download!"; \
        exit 1; \
    fi && \
    ls -lh model/

# Expose the port
EXPOSE 8080

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8080

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"] 