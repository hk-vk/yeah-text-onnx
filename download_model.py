"""
Model Downloader for Malayalam Text Classification API
------------------------------------------------------
This script checks if the ONNX model file exists in the model directory.
If not, it downloads the file from Google Drive.
"""

import os
import sys
import platform
import subprocess
import requests
from tqdm import tqdm
import shutil

# Google Drive file ID and destination path
DRIVE_FILE_ID = "1UfWvrS04ssG3WlDeV_uaqetrSZ70b6cW"
MODEL_DIR = "model"
MODEL_FILENAME = "malayalam_model1.onnx"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

def download_file_from_google_drive(file_id, destination):
    """
    Download a file from Google Drive using the direct download link.
    Shows a progress bar during download.
    """
    # Make sure the directory exists
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Google Drive download URL
    URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # First request to get the confirmation token
    session = requests.Session()
    response = session.get(URL, stream=True)
    
    # Check if there's a download warning (for large files)
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            URL = f"{URL}&confirm={value}"
            break
    
    # Get the file size if possible
    file_size = int(response.headers.get('content-length', 0))
    
    # Download with progress bar
    response = session.get(URL, stream=True)
    
    # Create a temporary file for downloading
    temp_file = f"{destination}.download"
    
    with open(temp_file, 'wb') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Downloading {MODEL_FILENAME}") as pbar:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    # Move the temp file to the final destination
    shutil.move(temp_file, destination)
    print(f"Download complete: {destination}")

def download_with_platform_command():
    """
    Download the model file using platform-specific commands as a fallback.
    """
    os_name = platform.system().lower()
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    print(f"Downloading model using {os_name} commands...")
    
    if os_name == "windows":
        # PowerShell command for Windows
        ps_command = f"""
        $url = "https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"
        $output = "{MODEL_PATH}"
        
        Write-Host "Downloading model file from Google Drive..."
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        
        $wc = New-Object System.Net.WebClient
        $wc.DownloadFile($url, $output)
        
        Write-Host "Download complete!"
        """
        
        # Save the PowerShell script to a temporary file
        ps_script_path = "download_model.ps1"
        with open(ps_script_path, "w") as f:
            f.write(ps_command)
        
        # Execute the PowerShell script
        subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", ps_script_path], check=True)
        
        # Clean up the temporary script
        os.remove(ps_script_path)
        
    elif os_name == "linux" or os_name == "darwin":  # Linux or macOS
        # Use curl or wget
        if shutil.which("curl"):
            subprocess.run([
                "curl", "-L", 
                f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}",
                "-o", MODEL_PATH
            ], check=True)
        elif shutil.which("wget"):
            subprocess.run([
                "wget", "--no-check-certificate",
                f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}",
                "-O", MODEL_PATH
            ], check=True)
        else:
            print("Error: Neither curl nor wget is available. Please install one of them.")
            return False
    else:
        print(f"Unsupported operating system: {os_name}")
        return False
    
    return True

def check_and_download_model():
    """
    Check if the model file exists, and download it if it doesn't.
    Returns True if the model is available (either existed or was downloaded successfully).
    """
    # Check if model directory exists
    if not os.path.exists(MODEL_DIR):
        print(f"Creating model directory: {MODEL_DIR}")
        os.makedirs(MODEL_DIR)
    
    # Check if model file exists
    if os.path.exists(MODEL_PATH):
        print(f"Model file already exists: {MODEL_PATH}")
        return True
    
    print(f"Model file not found: {MODEL_PATH}")
    print(f"Downloading from Google Drive (ID: {DRIVE_FILE_ID})...")
    
    try:
        # Try using the Python method first
        download_file_from_google_drive(DRIVE_FILE_ID, MODEL_PATH)
        return True
    except Exception as e:
        print(f"Python download failed: {str(e)}")
        print("Trying platform-specific commands...")
        
        # Fall back to platform-specific commands
        return download_with_platform_command()

if __name__ == "__main__":
    print("Malayalam Text Classification Model Downloader")
    print("=" * 50)
    
    if check_and_download_model():
        print("Model is ready to use!")
    else:
        print("Failed to download the model. Please download it manually.")
        print(f"Google Drive URL: https://drive.google.com/file/d/{DRIVE_FILE_ID}/view?usp=sharing")
        print(f"Save it to: {MODEL_PATH}")
        sys.exit(1) 