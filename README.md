# Noisy Medical Document Images OCR Platform

An AI-powered document transcription application designed to extract clear digital text records from noisy, low-quality, or stained medical document images.

## Features
- **Automated Pipeline:** Automatic fetching of the `devp1866/noisy-medical-document-images-ocr` dataset context.
- **Computer Vision Denoising:** Utilizes adaptive thresholding and bilateral filtering to strip away background shadows and wrinkles.
- **Lightweight Deployment:** Runs completely on CPU without needing heavy TensorFlow binaries.

## How to Deploy on Streamlit Cloud

1. Push this entire folder structure to your public **GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. Click **New app**, select your repository, branch, and set the Main file path to `streamlit_app.py`.
4. *Optional:* Since `kagglehub` accesses public Kaggle files, it should download smoothly. If authentication issues arise for private metrics, add your `KAGGLE_USERNAME` and `KAGGLE_KEY` into the Streamlit Cloud **Advanced Settings -> Secrets** manager.
5. Click **Deploy!**