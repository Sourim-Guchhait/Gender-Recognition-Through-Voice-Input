# 🎤 Gender Recognition through Voice

This project detects gender (Male/Female) from voice recordings using machine learning.

## Features
- Upload `.wav` audio file and predict gender
- Record audio live using browser mic
- Displays predicted gender instantly

## Tech Stack
- Python
- Streamlit
- scikit-learn (Logistic Regression)
- MFCC Feature Extraction (librosa)

## Team Members
- Sourim (Team Lead)
- Ayan
- Tathakata
- Supratic
- Tushar


## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py

# Project Structure.......

voice-gender-detection/
│
├── app.py                # Streamlit web app
├── train_model.py        # Training the Logistic Regression model
├── extract_features.py   # Extract MFCCs and generate dataset
│
├── models/               # Saved model and scaler
│   ├── gender_classifier.pkl
│   └── scaler.pkl
│
├── data/
│   ├── raw_audio/        # Original audio files (wav)
│   └── voice.csv         # Final extracted features dataset
│
├── requirements.txt      # List of required Python packages
└── README.md             # (You're reading it!)



