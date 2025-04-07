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
import gdown
import shutil

# Google Drive file ID and destination path
DRIVE_FILE_ID = "1UfWvrS04ssG3WlDeV_uaqetrSZ70b6cW"
MODEL_DIR = "model"
MODEL_FILENAME = "malayalam_model1.onnx"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

def download_with_gdown(file_id, destination):
    """
    Download a file from Google Drive using gdown.
    """
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    url = f'https://drive.google.com/uc?id={file_id}'
    print(f"Attempting download with gdown from URL: {url}")
    gdown.download(url, destination, quiet=False, fuzzy=True)
    # Check if download was successful (gdown raises exceptions on failure)
    if os.path.exists(destination):
        print(f"gdown download successful: {destination}")
        return True
    else:
        print("gdown download failed.")
        return False

def download_file_from_google_drive_requests(file_id, destination):
    """
    Download a file from Google Drive using the direct download link via requests.
    Shows a progress bar during download.
    """
    # Re-import requests and tqdm here as they are only used in this fallback
    import requests
    from tqdm import tqdm

    # Make sure the directory exists
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    print(f"Attempting download with requests from URL: {URL}")

    session = requests.Session()
    response = session.get(URL, stream=True)

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            URL = f"{URL}&confirm={value}"
            break

    file_size = int(response.headers.get('content-length', 0))
    response = session.get(URL, stream=True) # Re-request with confirmation if needed

    if response.status_code != 200:
        print(f"Requests download failed: Status code {response.status_code}")
        response.raise_for_status() # Raise exception for bad status

    temp_file = f"{destination}.download"
    try:
        with open(temp_file, 'wb') as f:
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Downloading {MODEL_FILENAME} (requests)") as pbar:
                for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Verify file size if possible
        if file_size != 0 and os.path.getsize(temp_file) != file_size:
             raise IOError(f"File size mismatch: Expected {file_size}, got {os.path.getsize(temp_file)}")

        shutil.move(temp_file, destination)
        print(f"Requests download complete: {destination}")
        return True # Indicate success
    except Exception as e:
        print(f"Error during requests download or verification: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file) # Clean up temp file on error
        raise # Re-raise the exception to be caught by the caller

def download_with_platform_command():
    """
    Download the model file using platform-specific commands as a fallback.
    """
    os_name = platform.system().lower()
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    print(f"Attempting download using platform command ({os_name})...")
    
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
        try:
            subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", ps_script_path], check=True, capture_output=True)
            print("Platform command (PowerShell) download successful.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Platform command (PowerShell) failed: {e}")
            print(f"Stdout: {e.stdout.decode()}")
            print(f"Stderr: {e.stderr.decode()}")
            return False
        finally:
            # Clean up the temporary script
            if os.path.exists(ps_script_path):
                os.remove(ps_script_path)
        
    elif os_name == "linux" or os_name == "darwin":  # Linux or macOS
        # Use curl or wget
        if shutil.which("curl"):
            try:
                subprocess.run([
                    "curl", "-L",
                    f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}",
                    "-o", MODEL_PATH
                ], check=True, capture_output=True)
                print("Platform command (curl) download successful.")
                return True
            except subprocess.CalledProcessError as e:
                print(f"Platform command (curl) failed: {e}")
                print(f"Stdout: {e.stdout.decode()}")
                print(f"Stderr: {e.stderr.decode()}")
                if os.path.exists(MODEL_PATH): # Clean up potentially incomplete file
                    os.remove(MODEL_PATH)
                return False
        elif shutil.which("wget"):
            try:
                subprocess.run([
                    "wget", "--no-check-certificate", "--quiet", # Add quiet flag
                    f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}",
                    "-O", MODEL_PATH
                ], check=True, capture_output=True)
                print("Platform command (wget) download successful.")
                return True
            except subprocess.CalledProcessError as e:
                print(f"Platform command (wget) failed: {e}")
                print(f"Stdout: {e.stdout.decode()}")
                print(f"Stderr: {e.stderr.decode()}")
                if os.path.exists(MODEL_PATH): # Clean up potentially incomplete file
                    os.remove(MODEL_PATH)
                return False
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
    Tries gdown first, then requests, then platform commands.
    Returns True if the model is available.
    """
    if not os.path.exists(MODEL_DIR):
        print(f"Creating model directory: {MODEL_DIR}")
        os.makedirs(MODEL_DIR)

    if os.path.exists(MODEL_PATH):
        # Basic check: verify file size is greater than a few KB to avoid tiny incorrect files
        try:
            if os.path.getsize(MODEL_PATH) > 10 * 1024: # Check if > 10KB
                 print(f"Model file already exists and seems valid: {MODEL_PATH}")
                 return True
            else:
                print(f"Existing model file {MODEL_PATH} is too small ({os.path.getsize(MODEL_PATH)} bytes). Redownloading.")
                os.remove(MODEL_PATH) # Remove invalid small file
        except OSError as e:
             print(f"Could not access existing model file {MODEL_PATH}: {e}. Attempting redownload.")

    print(f"Model file not found or invalid: {MODEL_PATH}")
    print(f"Attempting download from Google Drive (ID: {DRIVE_FILE_ID})...")

    # Try gdown first
    try:
        print("Trying download with gdown...")
        if download_with_gdown(DRIVE_FILE_ID, MODEL_PATH):
            # Add size verification after gdown download
            if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 10 * 1024:
                 print("gdown download successful and file size looks reasonable.")
                 return True
            else:
                 print("gdown download completed, but file is missing or too small. Removing.")
                 if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
                 raise ValueError("gdown downloaded an invalid file.") # Force fallback
        else:
            # gdown indicated failure without raising an exception (shouldn't normally happen)
             raise RuntimeError("gdown reported failure.")
    except Exception as e_gdown:
        print(f"gdown download failed: {str(e_gdown)}")
        # Fall through to try requests

    # Try requests as fallback 1
    try:
        print("Trying download with requests...")
        if download_file_from_google_drive_requests(DRIVE_FILE_ID, MODEL_PATH):
             # Size check already included in the requests function
             print("requests download successful.")
             return True
        else:
             # Should not happen as requests function raises exceptions on failure
             raise RuntimeError("requests function returned False unexpectedly.")
    except Exception as e_req:
        print(f"requests download failed: {str(e_req)}")
        # Fall through to try platform commands

    # Try platform commands as fallback 2
    try:
        print("Trying download with platform-specific command...")
        if download_with_platform_command():
            # Add size verification after platform command download
            if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 10 * 1024:
                 print("Platform command download successful and file size looks reasonable.")
                 return True
            else:
                 print("Platform command download completed, but file is missing or too small. Removing.")
                 if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
                 return False # Indicate final failure
        else:
            print("Platform command download failed.")
            return False # Indicate final failure
    except Exception as e_plat:
        print(f"Platform command download encountered an unexpected error: {str(e_plat)}")
        return False # Indicate final failure

if __name__ == "__main__":
    print("Malayalam Text Classification Model Downloader")
    print("=" * 50)
    
    if check_and_download_model():
        print("Model is ready to use!")
    else:
        print("Failed to download the model using all methods. Please check the URL and network.")
        print(f"Google Drive URL: https://drive.google.com/file/d/{DRIVE_FILE_ID}/view?usp=sharing")
        print(f"Target path: {MODEL_PATH}")
        sys.exit(1) 