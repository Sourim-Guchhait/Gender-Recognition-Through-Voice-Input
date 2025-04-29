import streamlit as st
import numpy as np
import librosa
import pickle
import sounddevice as sd
import soundfile as sf
import io
import joblib

# Load model and scaler
model = joblib.load(open('models/gender_classifier.pkl', 'rb'))
scaler = joblib.load(open('models/scaler.pkl', 'rb'))

# Extract features (MFCC)
def extract_features(file):
    audio, sr = librosa.load(file, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed.reshape(1, -1)

# Record audio
def record_voice(duration=4, fs=16000):
    st.info(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    virtualfile = io.BytesIO()
    sf.write(virtualfile, recording, fs, format='WAV')
    virtualfile.seek(0)
    return virtualfile

# Streamlit app
st.set_page_config(page_title="Gender Recognition App", page_icon=":microphone:", layout="centered")
st.title("🎤 Gender Recognition App")
st.markdown("Upload or record your voice to predict gender!")

# Choose input
input_method = st.radio("Choose input method:", ("Upload your voice file (.wav)", "Record Voice 🎙️"))

audio_file = None

if input_method == "Upload your voice file (.wav)":
    audio_file = st.file_uploader("Upload a WAV file", type=['wav'])

elif input_method == "Record Voice 🎙️":
    if st.button('🎙️ Start Recording'):
        audio_file = record_voice()

# If audio is available
if audio_file:
    st.audio(audio_file, format='audio/wav')

    features = extract_features(audio_file)
    features_scaled = scaler.transform(features)

    with st.spinner('Analyzing your voice and predicting...'):
        prediction = model.predict(features_scaled)

    st.success('✅ Prediction Complete!')

    if prediction[0] == 'male':
        st.markdown("### 👨 Predicted Gender: **Male**")
    else:
        st.markdown("### 👩 Predicted Gender: **Female**")
