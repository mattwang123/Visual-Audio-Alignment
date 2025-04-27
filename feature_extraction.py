import os
import cv2
import numpy as np
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings

# Suppress annoying warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def extract_visual_features(video_path, target_fps=10, resize_dim=(160, 120)):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    skip_interval = max(1, int(round(original_fps / target_fps)))
    
    prev_gray = None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    motion_energy = np.zeros(total_frames // skip_interval + 1)
    energy_idx = 0
    
    for frame_idx in range(0, total_frames, skip_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize_dim)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=1,
                winsize=15, iterations=1,
                poly_n=3, poly_sigma=0.5,
                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
            )
            magnitude = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
            motion_energy[energy_idx] = np.mean(magnitude)
            energy_idx += 1
        prev_gray = gray
    
    cap.release()
    return motion_energy[:energy_idx]

def extract_audio_features(video_path, target_sr=16000, hop_length=1024):
    audio, _ = librosa.load(video_path, sr=target_sr, mono=True)
    chunk_size = 10 * target_sr
    features = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        stft = librosa.stft(chunk, n_fft=2048, hop_length=hop_length)
        mag = np.abs(stft)
        rms = librosa.feature.rms(S=mag, frame_length=2048, hop_length=hop_length)[0]
        features.append(rms)
    return np.concatenate(features)

def process_video(video_path, output_dir):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            visual_future = executor.submit(extract_visual_features, video_path)
            audio_future = executor.submit(extract_audio_features, video_path)
            
            visual_feat = visual_future.result()
            audio_feat = audio_future.result()
        
        min_len = min(len(visual_feat), len(audio_feat))
        visual_feat = visual_feat[:min_len]
        audio_feat = audio_feat[:min_len]
        
        np.save(os.path.join(output_dir, 'visual_features', f'{base_name}.npy'), visual_feat)
        np.save(os.path.join(output_dir, 'audio_features', f'{base_name}.npy'), audio_feat)

        return base_name  # Return for printing when done
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return None

def batch_process(dataset_dir, output_dir, max_workers=4):
    video_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    os.makedirs(os.path.join(output_dir, 'visual_features'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'audio_features'), exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_video, os.path.join(dataset_dir, video_file), output_dir): video_file for video_file in video_files}
        
        with tqdm(total=len(futures), desc="Processing videos", ncols=100) as pbar:
            for future in as_completed(futures):
                video_file = futures[future]
                result = future.result()
                if result is not None:
                    tqdm.write(f"Finished processing: {result}")
                pbar.update(1)
