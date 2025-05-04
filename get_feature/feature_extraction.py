import os
import cv2
import numpy as np
import librosa
import warnings
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

warnings.filterwarnings("ignore")

# === Setup Logging ===
log_dir = r"D:\lenovo\mia_final_project\logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.info("🚀 Starting batch processing...")

# === Feature Extraction ===
def extract_visual_features(video_path, target_fps=5, resize_dim=(96, 96)):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps == 0 or not cap.isOpened():
        return np.empty((0,))

    skip = max(1, round(original_fps / target_fps))
    features = []

    for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), skip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize_dim)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features.append(gray.flatten())

    cap.release()
    return np.array(features)

def extract_audio_features(video_path, sr=16000, n_mfcc=13):
    try:
        y, _ = librosa.load(video_path, sr=sr)
        if len(y) == 0:
            return np.empty((0, 13))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfcc.T  # shape: (time, n_mfcc)
    except Exception:
        return np.empty((0, 13))

def parse_label_from_filename(filename):
    if 'good' in filename.lower():
        return 1
    elif 'bad' in filename.lower():
        return 0
    return -1  # unknown/ignore

def process_video(video_path, output_dir):
    filename = os.path.basename(video_path)
    name, _ = os.path.splitext(filename)
    label = parse_label_from_filename(name)

    if label == -1:
        logging.info(f"Skipping {filename}, label unknown.")
        return None

    try:
        visual = extract_visual_features(video_path)
        audio = extract_audio_features(video_path)

        if visual.ndim != 2 or audio.ndim != 2 or len(visual) == 0 or len(audio) == 0:
            logging.warning(f"Skipping {video_path} due to invalid shape: visual {visual.shape}, audio {audio.shape}")
            return None

        min_len = min(len(visual), len(audio))
        visual, audio = visual[:min_len], audio[:min_len]

        save_path = os.path.join(output_dir, f"{name}.npz")
        np.savez_compressed(save_path, visual=visual, audio=audio, label=label)
        logging.info(f"✅ Saved features: {save_path}")
        return name

    except Exception as e:
        logging.error(f"❌ Error processing {video_path}: {str(e)}", exc_info=True)
        return None

def batch_process(dataset_dir, output_dir, max_workers=4):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(dataset_dir) if f.endswith(('.mp4', '.avi'))]

    if not files:
        logging.warning(f"No video files found in: {dataset_dir}")
        return

    logging.info(f"Found {len(files)} video files in {dataset_dir}")
    processed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_video, os.path.join(dataset_dir, f), output_dir): f
            for f in files
        }

        with tqdm(total=len(futures), desc="Extracting", ncols=100) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    processed_count += 1
                    if processed_count % 100 == 0:
                        logging.info(f"✅ {processed_count} videos processed so far...")
                pbar.update(1)

    logging.info(f"🎉 Done! Total videos processed: {processed_count}")
