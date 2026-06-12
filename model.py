import easyocr
import cv2
import numpy as np
from PIL import Image

class MedicalOCRModel:
    def __init__(self):
        # Initialize the EasyOCR reader for English text
        # gpu=False ensures compatibility with standard Streamlit Cloud containers
        self.reader = easyocr.Reader(['en'], gpu=False)

    def preprocess_image(self, image: Image.Image):
        """
        Preprocesses noisy medical images to improve OCR accuracy.
        Converts to grayscale and applies adaptive thresholding to remove noise.
        """
        # Convert PIL Image to OpenCV format (BGR)
        open_cv_image = np.array(image)
        if len(open_cv_image.shape) == 3:
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            
        # Convert to grayscale
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to remove noise while keeping edges sharp
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Binary thresholding to make text stand out sharply
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh

    def extract_text(self, image: Image.Image):
        """
        Preprocesses the image and performs OCR extraction.
        Returns a list of dictionaries containing text and confidence scores.
        """
        processed_img = self.preprocess_image(image)
        
        # Perform OCR
        results = self.reader.readtext(processed_img)
        
        extracted_data = []
        for (bbox, text, prob) in results:
            if prob > 0.20: # Filter out extremely low-confidence noise
                extracted_data.append({
                    "text": text,
                    "confidence": float(prob)
                })
                
        return extracted_data