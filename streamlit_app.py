import streamlit as st
from PIL import Image
import os
from data_loader import download_medical_dataset
from model import MedicalOCRModel

# Set page configuration
st.set_page_config(page_title="Medical OCR Digitizer", page_icon="🏥", layout="centered")

# App header
st.title("🏥 Noisy Medical Document OCR Detection")
st.write("Convert noisy, blurry, or scanned physical medical documents into clean, digital text.")

# 1. Trigger automated dataset download/verification in the background
@st.cache_resource
def initialize_system():
    # Cache downloading so it only runs once when the app boots up
    dataset_path = download_medical_dataset()
    ocr_model = MedicalOCRModel()
    return dataset_path, ocr_model

with st.spinner("Initializing AI Models and verifying Kaggle environments..."):
    dataset_path, model = initialize_system()

if dataset_path:
    
else:
    st.sidebar.warning("⚠️ Running in Standalone Mode (Kaggle Dataset Unreachable)")

# 2. File Upload UI Section
st.subheader("Upload Medical Document")
uploaded_file = st.file_uploader("Choose a scanned image (JPG, PNG, JPEG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Document", use_container_width=True)
        
    with col2:
        st.write("🔄 **Processing Engine Status:** Ready")
        if st.button("Extract Text from Document", type="primary"):
            with st.spinner("Denoising image and running OCR analysis..."):
                # Run OCR model pipeline
                extracted_results = model.extract_text(image)
                
            st.success("Analysis Complete!")
            
            # Display Extracted Text Output
            st.subheader("Extracted Digital Text:")
            if extracted_results:
                full_text = ""
                for item in extracted_results:
                    full_text += item['text'] + "\n"
                    # Display individual blocks with confidence tooltips
                    st.text(f"📝 {item['text']} (Conf: {item['confidence']:.2%})")
                
                # Add a convenient download button for the generated text
                st.download_button(
                    label="Download Clean Text File",
                    data=full_text,
                    file_name="extracted_medical_record.txt",
                    mime="text/plain"
                )
            else:
                st.info("No legible text detected. The document might be too noisy or blank.")
