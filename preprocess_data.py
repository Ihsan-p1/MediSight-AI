import os
import zipfile
import shutil
from pathlib import Path

# =============================================================
# CONFIGURATION
# =============================================================

RAW_DIR = Path("datasets")
PROCESSED_DIR = Path("data")

# Dataset Mappings (Source in 'datasets/' -> Target in 'data/')
DATASET_MAP = {
    "face": "wider-face-torchvision-compatible", # Zip file
    "landmarks": "JDAI-CV_lapa-dataset",         # Zip file
    "rppg": "UBFC",                              # Folder
    "emotion": "fer2013",                        # Folder (train/test)
    "fatigue": "Driver Drowsiness Dataset (DDD)",# Folder
    "pain": "CK+48"                              # Folder
}

# =============================================================
# HELPERS
# =============================================================

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)

def unzip_file(zip_path, extract_to):
    print(f"Unzipping {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Done.")
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is a bad zip file.")
    except Exception as e:
        print(f"Error unzipping {zip_path}: {e}")

def organize_face():
    print("\n--- Processing Face Data (WIDER FACE) ---")
    target_dir = PROCESSED_DIR / "face"
    ensure_dir(target_dir)
    
    # Check for extracted folder first (Kaggle CLI often extracts automatically)
    source_folder = RAW_DIR / "widerface"
    if source_folder.exists():
        print(f"Copying from {source_folder}...")
        # Use copytree with dirs_exist_ok to merge if needed
        shutil.copytree(source_folder, target_dir, dirs_exist_ok=True)
        return

    # Fallback to zip
    zip_name = "wider-face-torchvision-compatible.zip"
    zip_path = RAW_DIR / zip_name
    
    if zip_path.exists():
        # Check if already extracted
        if not (target_dir / "WIDER_train").exists():
            unzip_file(zip_path, target_dir)
        else:
            print("Skipping unzip: WIDER_train already exists.")
    else:
        print(f"Warning: {zip_name} or 'widerface' folder not found.")

def organize_landmarks():
    print("\n--- Processing Landmark Data (LaPa) ---")
    target_dir = PROCESSED_DIR / "landmarks"
    ensure_dir(target_dir)
    
    zip_name = "JDAI-CV_lapa-dataset.zip"
    zip_path = RAW_DIR / zip_name
    
    if zip_path.exists():
        if not (target_dir / "Lapa").exists():
             unzip_file(zip_path, target_dir)
        else:
             print("Skipping unzip: Lapa folder already exists.")
    else:
        print(f"Warning: {zip_name} not found.")

def organize_rppg():
    print("\n--- Processing rPPG Data (UBFC) ---")
    target_dir = PROCESSED_DIR / "rppg"
    ensure_dir(target_dir)
    
    source_dir = RAW_DIR / "UBFC"
    if source_dir.exists():
        # UBFC structure is often messy. Just copy for now.
        # If it's huge, maybe move instead of copy? Let's copy to be safe.
        print(f"Copying {source_dir} to {target_dir}...")
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    else:
        print("Warning: UBFC folder not found.")

def organize_emotion():
    print("\n--- Processing Emotion Data (FER2013) ---")
    target_dir = PROCESSED_DIR / "emotion"
    ensure_dir(target_dir)
    
    # FER2013 usually extracts to 'train' and 'test' directly in datasets/
    source_train = RAW_DIR / "train"
    source_test = RAW_DIR / "test"
    
    if source_train.exists():
        print("Moving 'train' folder...")
        shutil.move(str(source_train), str(target_dir / "train"))
    
    if source_test.exists():
        print("Moving 'test' folder...")
        shutil.move(str(source_test), str(target_dir / "test"))
        
    if not source_train.exists() and not (target_dir / "train").exists():
        print("Warning: FER2013 'train' folder not found.")

def organize_fatigue():
    print("\n--- Processing Fatigue Data (DDD) ---")
    target_dir = PROCESSED_DIR / "fatigue"
    ensure_dir(target_dir)
    
    # Check for nested folder
    nested_dir = target_dir / "Driver Drowsiness Dataset (DDD)"
    if nested_dir.exists():
        print(f"Flattening directory structure from {nested_dir}...")
        for item in nested_dir.iterdir():
            shutil.move(str(item), str(target_dir))
        # Remove empty nested dir
        nested_dir.rmdir()
    
    # Check if Drowsy/Non Drowsy exist
    if (target_dir / "Drowsy").exists() and (target_dir / "Non Drowsy").exists():
        print("Fatigue data organized successfully (Drowsy/Non Drowsy).")
    else:
        print("Warning: Drowsy/Non Drowsy folders not found.")

def organize_pain():
    print("\n--- Processing Pain Data (CK+ Proxy) ---")
    target_dir = PROCESSED_DIR / "pain"
    ensure_dir(target_dir)
    
    # We want to filter for specific emotions that resemble pain
    # Pain-relevant AUs are often found in: Disgust, Sadness, Surprise (shock)
    target_classes = ["disgust", "sadness", "surprise"]
    
    # Create a 'processed' folder for the selected proxies
    processed_pain_dir = target_dir / "processed_proxy"
    ensure_dir(processed_pain_dir)
    
    found_any = False
    for class_name in target_classes:
        source_class_dir = target_dir / class_name
        if source_class_dir.exists():
            print(f"Found proxy class: {class_name}")
            # Copy to processed folder
            dest = processed_pain_dir / class_name
            if not dest.exists():
                shutil.copytree(source_class_dir, dest)
            found_any = True
        else:
            print(f"Warning: Class {class_name} not found in CK+.")
            
    if found_any:
        print(f"Pain proxy data prepared in {processed_pain_dir}")
    else:
        print("Warning: No suitable proxy classes found in CK+.")

# =============================================================
# MAIN
# =============================================================

def main():
    print("Starting Data Preprocessing...")
    ensure_dir(PROCESSED_DIR)
    
    organize_face()
    organize_landmarks()
    organize_rppg()
    organize_emotion()
    organize_fatigue()
    organize_pain()
    
    print("\nPreprocessing Complete!")
    print(f"Data organized in: {PROCESSED_DIR.absolute()}")

if __name__ == "__main__":
    main()
