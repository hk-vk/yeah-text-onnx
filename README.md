# Malayalam Fake News Detection API

A FastAPI-based API for Malayalam text classification using an ONNX model to detect fake news.

## Features

- ONNX-based text classification for Malayalam language
- Memory-optimized inference for large models
- RESTful API with FastAPI
- Swagger UI documentation
- Automatic model download from Google Drive

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yeah-text-onnx.git
cd yeah-text-onnx
```

2. Create a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Model files:
   - The ONNX model file will be automatically downloaded from Google Drive when you start the server
   - If you prefer to download it manually, get it from [this Google Drive link](https://drive.google.com/file/d/1UfWvrS04ssG3WlDeV_uaqetrSZ70b6cW/view?usp=sharing) and place it at `model/malayalam_model1.onnx`
   - Make sure the tokenizer files are in the `model` directory

## Running the API

Start the server using the provided script:
```bash
python start_server.py
```

The API will be available at:
- Main API: http://127.0.0.1:8080
- Documentation: http://127.0.0.1:8080/docs
- ReDoc: http://127.0.0.1:8080/redoc

## API Endpoints

- `GET /`: Health check endpoint
- `POST /predict`: Prediction endpoint
  - Request body: `{"text": "your malayalam text here"}`
  - Response: `{"predicted_class": 0, "confidence": 0.95}`

## Testing the API

Use the provided test script:
```bash
python test_api.py
```

## Manual Model Download

If the automatic download fails, you can manually download the model:

1. Download the model from [Google Drive](https://drive.google.com/file/d/1UfWvrS04ssG3WlDeV_uaqetrSZ70b6cW/view?usp=sharing)
2. Create a `model` directory if it doesn't exist
3. Place the downloaded file in the `model` directory as `malayalam_model1.onnx`

## Notes

- The large model file (906 MB) is not stored in the Git repository
- The model is automatically downloaded from Google Drive when needed
- Memory optimization is applied to handle large model files efficiently 