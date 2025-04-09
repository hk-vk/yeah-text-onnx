import subprocess
import sys
import os
import time
import webbrowser
from threading import Thread

def run_streamlit():
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])

def run_fastapi():
    subprocess.run([sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"])

if __name__ == "__main__":
    # Start FastAPI server in a separate thread
    fastapi_thread = Thread(target=run_fastapi)
    fastapi_thread.daemon = True
    fastapi_thread.start()
    
    # Wait a moment for the API to start
    time.sleep(2)
    
    # Open Streamlit in the default browser
    webbrowser.open("http://localhost:8501")
    
    # Run Streamlit in the main thread
    run_streamlit() 