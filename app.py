import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
import time

# Setup paths
OUTPUT_DIR = "temp_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FILENAME = os.path.join(OUTPUT_DIR, "recorded.wav")

st.set_page_config(page_title="Voice Gender Detection")
st.title("Gender Detection from Voice")
st.write("Click below to record your voice and see the prediction!")

# UI Settings
duration = st.slider("Recording Duration (seconds)", 2, 10, 4)

# Record Button
if st.button("Record"):
    st.info("Recording...")
    fs = 16000  # 16kHz
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        write(FILENAME, fs, recording)
        st.success(f" Recording saved: {FILENAME}")
    except Exception as e:
        st.error(f"Recording failed: {e}")

# Dummy Prediction Button
if st.button("Predict Gender"):
    if os.path.exists(FILENAME):
        st.success(" Predicted Gender: **Male**")  # Here is the Dummy Output
    else:
        st.warning(" Please record your voice first.")
