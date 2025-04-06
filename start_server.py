"""
Malayalam Fake News Detection API Server Launcher
------------------------------------------------
This script starts the FastAPI server on the specified host and port
"""

import os
import sys
import subprocess

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")  # Use environment variable or default to 0.0.0.0
PORT = int(os.getenv("PORT", 8080))  # Use environment variable or default to 8080

def start_server():
    """Start the FastAPI server"""
    print(f"Starting Malayalam Fake News Detection API on http://{HOST}:{PORT}")
    print("=" * 60)
    print("API will be available at:")
    print(f"  - Main API:     http://{HOST}:{PORT}")
    print(f"  - Documentation: http://{HOST}:{PORT}/docs")
    print(f"  - ReDoc:        http://{HOST}:{PORT}/redoc")
    print("=" * 60)
    print("Press CTRL+C to stop the server")
    print("=" * 60)
    
    # Use uvicorn to start the server
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "main:app", 
        "--host", HOST, 
        "--port", str(PORT),
        "--reload"
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServer stopped")

if __name__ == "__main__":
    # Check if main.py exists
    if not os.path.exists("main.py"):
        print("Error: main.py not found in the current directory")
        sys.exit(1)
        
    # Start the server
    start_server()