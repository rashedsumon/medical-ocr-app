import os
import kagglehub

def download_medical_dataset():
    """
    Automatically downloads the latest version of the noisy medical document 
    images OCR dataset from Kaggle using kagglehub.
    """
    try:
        print("Checking/Downloading dataset from Kaggle...")
        # Download latest version
        path = kagglehub.dataset_download("devp1866/noisy-medical-document-images-ocr")
        print(f"Path to dataset files verified at: {path}")
        return path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    # Test script locally
    download_medical_dataset()