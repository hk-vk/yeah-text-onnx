import requests
import json

# API endpoint configuration
HOST = "127.0.0.1"
PORT = 8080
BASE_URL = f"http://{HOST}:{PORT}"

def test_health():
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/")
        print("\nHealth Check Response:")
        print(json.dumps(response.json(), indent=2))
    except requests.exceptions.ConnectionError:
        print(f"\nError: Could not connect to the server. Make sure it's running on http://{HOST}:{PORT}")

def test_prediction(text):
    """Test the prediction endpoint"""
    url = f"{BASE_URL}/predict"
    
    try:
        # Prepare the request data
        data = {
            "text": text
        }
        
        # Make the POST request
        response = requests.post(url, json=data)
        
        print(f"\nPrediction for text: {text}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        
        return response.json()
    except requests.exceptions.ConnectionError:
        print(f"\nError: Could not connect to the server. Make sure it's running on http://{HOST}:{PORT}")
    except Exception as e:
        print(f"\nError making prediction: {str(e)}")

if __name__ == "__main__":
    print("Testing Malayalam Fake News Detection API")
    print("========================================")
    print("Make sure the server is running with:")
    print(f"uvicorn main:app --reload --host 0.0.0.0 --port {PORT}")
    print("========================================")
    
    # Test the health endpoint
    test_health()
    
    # Test some Malayalam text examples
    examples = [
        "കേരളത്തിൽ കനത്ത മഴയ്ക്ക് സാധ്യത",  # Weather news
        "ഇന്ന് രാവിലെ 10 മണിക്ക് സ്കൂൾ അടച്ചു",  # School closure news
        "അത്ഭുതകരമായ വാർത്ത! ഞെട്ടിക്കുന്ന വെളിപ്പെടുത്തൽ",  # Clickbait example
    ]
    
    print("\nTesting multiple examples:")
    for text in examples:
        test_prediction(text)
        print("-" * 50) 