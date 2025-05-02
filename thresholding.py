import os
import cv2
import numpy as np

dataset_dir = 'D:\lenovo\mia_final_project\Visual-Audio-Alignment\data_samples'
#'/Users/ruyutong/Visual-Audio-Alignment/data_samples' 
audio_feature_dir = os.path.join(dataset_dir, 'audio_features')
visual_feature_dir = os.path.join(dataset_dir, 'visual_features')
output_dir = dataset_dir  # or use a separate folder if preferred

def threshold_anomaly_detection(audio_feature, visual_feature):
    discrepancy = np.abs(audio_feature - visual_feature)
    threshold = np.mean(discrepancy) + 2 * np.std(discrepancy)
    anomalies = discrepancy > threshold
    return anomalies.astype(int)

def draw_blinking_alert(frame, blink_on, color=(0, 0, 255), thickness=5):
    if blink_on:
        h, w, _ = frame.shape
        return cv2.rectangle(frame.copy(), (0, 0), (w-1, h-1), color, thickness)
    return frame

# Process each video in dataset_dir
for filename in os.listdir(dataset_dir):
    if filename.endswith('.mp4'):
        base_name = os.path.splitext(filename)[0]
        video_path = os.path.join(dataset_dir, filename)

        audio_path = os.path.join(audio_feature_dir, f"{base_name}.npy")
        visual_path = os.path.join(visual_feature_dir, f"{base_name}.npy")

        if not os.path.exists(audio_path) or not os.path.exists(visual_path):
            print(f"Skipping {filename} — missing audio/visual features.")
            continue

        # Load features
        audio_feature = np.load(audio_path)
        visual_feature = np.load(visual_path)

        # Ensure length matches the video frame count
        anomalies = threshold_anomaly_detection(audio_feature, visual_feature)

        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = os.path.join(output_dir, f"{base_name}_blink.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_index >= len(anomalies):
                break

            if anomalies[frame_index] == 1:
                blink_on = (frame_index // 5) % 2 == 0
                frame = draw_blinking_alert(frame, blink_on)

            out.write(frame)
            frame_index += 1

        cap.release()
        out.release()
        print(f"Processed: {filename} → {output_path}")


