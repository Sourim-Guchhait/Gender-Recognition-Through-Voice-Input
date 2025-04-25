import librosa
import numpy as np
import pandas as pd
import os
from spafe.features.lpcc import lpcc as spafe_lpcc
from spafe.utils.lpc import lpc as spafe_lpc

def extract_features(audio_path, sr=22050, n_mfcc=20, lpcc_order=12, n_lpcc=13):
    """Extracts MFCCs, spectral centroid, spectral bandwidth, and LPCCs from an audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    # Aggregate MFCCs
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    # Aggregate Spectral Features
    centroid_mean = np.mean(spectral_centroid)
    centroid_std = np.std(spectral_centroid)
    bandwidth_mean = np.mean(spectral_bandwidth)
    bandwidth_std = np.std(spectral_bandwidth)

    # Extract LPCCs
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length, axis=0)

    if frames.shape[0] < lpcc_order + 1:
        print(f"Warning: Frame length too short for LPC order in {audio_path}. Skipping LPCC.")
        lpccs_mean = np.zeros(n_lpcc)
        lpccs_std = np.zeros(n_lpcc)
    else:
        try:
            lpc_coeffs = np.apply_along_axis(func1d=spafe_lpc, axis=1, arr=frames, order=lpcc_order)
            lpcc_coeffs = spafe_lpcc(lpc_coeffs, num_ceps=n_lpcc)
            lpccs_mean = np.mean(lpcc_coeffs, axis=0)
            lpccs_std = np.std(lpcc_coeffs, axis=0)
        except Exception as e:
            print(f"Error extracting LPCCs from {audio_path}: {e}")
            lpccs_mean = np.zeros(n_lpcc)
            lpccs_std = np.zeros(n_lpcc)

    feature_vector = np.concatenate([mfccs_mean, mfccs_std,
                                       [centroid_mean, centroid_std,
                                        bandwidth_mean, bandwidth_std],
                                       lpccs_mean, lpccs_std])
    return feature_vector

def process_data(audio_folder, n_mfcc=20, lpcc_order=12, n_lpcc=13):
    """Processes audio files, extracts features (including LPCCs), and returns features with labels."""
    features = []
    labels = []
    for gender_folder in os.listdir(audio_folder):
        gender_path = os.path.join(audio_folder, gender_folder)
        if os.path.isdir(gender_path) and gender_folder in ['male', 'female']:
            gender_label = 1 if gender_folder == 'male' else 0
            for filename in os.listdir(gender_path):
                if filename.endswith('.wav'):
                    audio_path = os.path.join(gender_path, filename)
                    feature_vector = extract_features(audio_path, n_mfcc=n_mfcc,
                                                     lpcc_order=lpcc_order, n_lpcc=n_lpcc)
                    if feature_vector is not None:
                        features.append(feature_vector)
                        labels.append(gender_label)
    return features, labels

if __name__ == "__main__":
    audio_data_folder = 'raw_audio'  # Replace with your audio data folder
    n_mfcc_coeffs = 20
    lpcc_order = 12
    n_lpcc_coeffs = 13
    extracted_features, labels = process_data(audio_data_folder, n_mfcc=n_mfcc_coeffs,
                                             lpcc_order=lpcc_order, n_lpcc=n_lpcc_coeffs)

    if extracted_features:
        # Create feature names
        mfcc_names_mean = [f'mfcc_mean_{i}' for i in range(n_mfcc_coeffs)]
        mfcc_names_std = [f'mfcc_std_{i}' for i in range(n_mfcc_coeffs)]
        spectral_names = ['centroid_mean', 'centroid_std', 'bandwidth_mean', 'bandwidth_std']
        lpcc_names_mean = [f'lpcc_mean_{i}' for i in range(n_lpcc_coeffs)]
        lpcc_names_std = [f'lpcc_std_{i}' for i in range(n_lpcc_coeffs)]

        feature_names = mfcc_names_mean + mfcc_names_std + spectral_names + lpcc_names_mean + lpcc_names_std
        df = pd.DataFrame(extracted_features, columns=feature_names)
        df['label'] = labels

        csv_filename = 'voice_features_with_lpcc.csv'
        df.to_csv(csv_filename, index=False)
        print(f"Features (including LPCCs) and labels saved to {csv_filename}")
    else:
        print("No features were extracted.")
