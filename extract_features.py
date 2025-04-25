import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Set the folder where .wav files are
AUDIO_DIR = os.path.join("data", "raw_audio/data")
OUTPUT_CSV = os.path.join("data", "voice.csv")

# Function to extract MFCCs
def extract_mfcc(filepath, n_mfcc=13):
    try:
        y, sr = librosa.load(filepath, sr=16000)  # Load at 16kHz
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)  # Average over time
        return mfcc_mean
    except Exception as e:
        print(f" Error processing {filepath}: {e}")
        return None

# Loop through all wav files
features = []
files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]

print(f" Found {len(files)} audio files to process.\n")

for file in tqdm(files, desc="Extracting MFCCs"):
    path = os.path.join(AUDIO_DIR, file)
    label = file.split("_")[0].lower()  # e.g., "male" from "male_01.wav"
    mfccs = extract_mfcc(path)
    if mfccs is not None:
        features.append(np.append(mfccs, [label, file]))

# Define column names
columns = [f"mfcc{i+1}" for i in range(13)] + ["label", "filename"]

# Save to CSV
df = pd.DataFrame(features, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n Done! Features saved to: {OUTPUT_CSV}")

