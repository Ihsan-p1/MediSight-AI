import os
import subprocess
import requests
import sys

# =============================================================
# HELPERS
# =============================================================

def download_kaggle(dataset, out_dir="datasets"):
    """
    dataset: 'owner/dataset-name'
    example: 'berkerisen/ubfcrppg-dataset'
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Check if already downloaded (simple check: look for folder with dataset name or zip)
    # Kaggle datasets usually extract to a folder or files. 
    # This is a heuristic.
    dataset_name = dataset.split("/")[-1]
    
    # Explicit mapping for known datasets that extract to different names
    folder_map = {
        "jawadulkarim117/rppg-heart-rate": "UBFC",
        "msambare/fer2013": "train", # FER extracts to train/test
        "ismailnasri20/driver-drowsiness-dataset-ddd": "Driver Drowsiness Dataset (DDD)",
        "dhruv4930/wider-face-torchvision-compatible": "widerface"
    }
    
    check_name = folder_map.get(dataset, dataset_name)
    possible_dir = os.path.join(out_dir, check_name)
    possible_zip = os.path.join(out_dir, dataset_name + ".zip")
    
    if os.path.exists(possible_dir) or os.path.exists(possible_zip):
        print(f"Skipping {dataset}: Already exists at {possible_dir} or {possible_zip}")
        return

    print(f"Downloading Kaggle dataset: {dataset}")
    # Check if kaggle is installed and find the executable
    kaggle_exe = "kaggle"
    try:
        subprocess.run(["kaggle", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Try to find it in the user scripts directory
        # Based on debug output: C:\Users\Lenovo\AppData\Roaming\Python\Python314\site-packages
        # Scripts should be at: C:\Users\Lenovo\AppData\Roaming\Python\Python314\Scripts
        user_scripts_dir = os.path.expanduser("~\\AppData\\Roaming\\Python\\Python314\\Scripts")
        possible_exe = os.path.join(user_scripts_dir, "kaggle.exe")
        
        if os.path.exists(possible_exe):
            # print(f"Debug: Found kaggle at {possible_exe}")
            kaggle_exe = possible_exe
        else:
            print(f"Error: 'kaggle' command not found in PATH and not found at {possible_exe}.")
            print("Please ensure 'kaggle.exe' is in your PATH or install it via 'pip install kaggle'.")
            return

    # Construct command
    cmd = [kaggle_exe, "datasets", "download", "-d", dataset, "-p", out_dir, "--unzip"]
    
    # Run command
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"Successfully downloaded {dataset}")
    else:
        print(f"Failed to download {dataset}")


def download_github(repo, out_dir="datasets"):
    """
    repo: 'owner/repo'
    example: 'tdencker/cohface'
    """
    os.makedirs(out_dir, exist_ok=True)
    safe_name = repo.replace("/", "_")
    path = os.path.join(out_dir, safe_name + ".zip")

    if os.path.exists(path):
        print(f"Skipping {repo}: Already exists at {path}")
        return

    url = f"https://github.com/{repo}/archive/refs/heads/master.zip"
    print(f"Downloading GitHub repo: {repo}")
    
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Saved:", path)
        print("Note: You may need to unzip this file manually or add unzip logic.")
    except Exception as e:
        print(f"Failed to download {repo}: {e}")

# =============================================================
# MAIN EXECUTION
# =============================================================

def download_all():
    print("Starting MediSight-AI Dataset Download...")
    print("Ensure you have your 'kaggle.json' API token set up correctly.")
    
    # 1. Face Detection: WIDER FACE (Alternative)
    # Original: xiyanghu/wider-face-dataset (Deleted)
    # New: dhruv4930/wider-face-torchvision-compatible
    download_kaggle("dhruv4930/wider-face-torchvision-compatible")

    # 2. Face Landmarks: LaPa (GitHub)
    # Reason: 106 landmarks, high quality, scriptable.
    download_github("JDAI-CV/lapa-dataset")

    # 3. rPPG: rPPG Heart Rate (Alternative)
    # Original: berkerisen/ubfcrppg-dataset (Deleted)
    # New: jawadulkarim117/rppg-heart-rate
    download_kaggle("jawadulkarim117/rppg-heart-rate")

    # 4. Emotion: FER2013 (Kaggle)
    # Reason: Standard dataset for basic emotions.
    download_kaggle("msambare/fer2013")

    # 5. Fatigue: Driver Drowsiness Dataset (Alternative)
    # Original: serkanpaci/yawdd (Deleted)
    # New: ismailnasri20/driver-drowsiness-dataset-ddd
    download_kaggle("ismailnasri20/driver-drowsiness-dataset-ddd")

    # 6. Pain (Proxy): CK+ (Kaggle)
    # Reason: UNBC-McMaster requires manual agreement. CK+ is a standard alternative 
    # that includes "Disgust" and "Sadness" which share Action Units with pain.
    download_kaggle("shawon10/ckplus")

    print("\n--- Download Summary ---")
    print("All datasets have been processed.")
    print("Note: CK+ is used as a proxy for Pain features due to access restrictions on UNBC-McMaster.")

if __name__ == "__main__":
    download_all()
