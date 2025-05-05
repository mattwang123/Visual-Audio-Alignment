import os
import numpy as np

# === Path setup ===
preprocessed_dir = r"D:\lenovo\mia_final_project\Visual-Audio-Alignment\preprocessed_output"
visual_dir = r"D:\lenovo\mia_final_project\Visual-Audio-Alignment\visual_features"
audio_dir = r"D:\lenovo\mia_final_project\Visual-Audio-Alignment\audio_features"   

os.makedirs(visual_dir, exist_ok=True)
os.makedirs(audio_dir, exist_ok=True)

# === Process and save ===
for filename in os.listdir(preprocessed_dir):
    if not filename.endswith(".npz"):
        continue

    source_path = os.path.join(preprocessed_dir, filename)
    try:
        arr = np.load(source_path)

        visual = arr['visual']
        audio = arr['audio']

        # Skip malformed or 1D data
        if visual.ndim != 2 or audio.ndim != 2 or len(visual) == 0 or len(audio) == 0:
            print(f"Skipping due to invalid shape: {filename}")
            continue

        # Save visual only
        visual_path = os.path.join(visual_dir, filename.replace(".npz", "_visual.npz"))
        np.savez_compressed(visual_path, visual=visual)

        # Save audio only
        audio_path = os.path.join(audio_dir, filename.replace(".npz", "_audio.npz"))
        np.savez_compressed(audio_path, audio=audio)

    except Exception as e:
        print(f"Skipping {filename} due to error: {e}")