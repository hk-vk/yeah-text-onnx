FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    git-lfs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Initialize Git LFS (needed inside the container)
RUN git lfs install --system

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
# Note: This COPY command needs the repository context to have LFS pointers
COPY . .

# Pull the LFS files after copying the repo contents
# This assumes the build context includes the .git directory and LFS pointers
RUN git lfs pull

# Create model directory (if needed, though LFS should place it correctly)
# RUN mkdir -p model # This might not be necessary if LFS places the file correctly

# Verify model file exists after LFS pull
RUN if [ ! -f model/malayalam_model1.onnx ]; then echo "ERROR: LFS file model/malayalam_model1.onnx not found after pull!"; exit 1; fi
RUN ls -lh model/ # Optional: List contents to check file size in build logs

# Expose the port
EXPOSE 8080

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8080

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"] 