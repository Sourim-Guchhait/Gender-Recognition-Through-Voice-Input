# rename_audio_files.py
import os

RAW_AUDIO_DIR = "data/raw_audio/data"
categories = ["male", "female"]

for category in categories:
    folder_path = os.path.join(RAW_AUDIO_DIR, category)
    if not os.path.exists(folder_path):
        continue

    files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    for i, file in enumerate(files, 1):
        old_path = os.path.join(folder_path, file)
        new_filename = f"{category}_{i:02d}.wav"
        new_path = os.path.join(RAW_AUDIO_DIR, new_filename)

        # Rename by moving file to parent directory with new name
        os.rename(old_path, new_path)
        print(f"Renamed: {file} → {new_filename}")

