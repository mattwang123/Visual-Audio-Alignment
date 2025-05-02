import os
import random
import pandas as pd
import subprocess
import numpy as np
from pydub import AudioSegment
from pydub.generators import WhiteNoise
from pydub.effects import speedup

def generate_bad_samples(df: pd.DataFrame, video_dir: str, output_dir: str, output_csv: str) -> None:
    """
    Generate bad samples by cutting good samples into pieces and applying misalignments.
    
    Args:
        df: DataFrame with columns ['VideoID', 'StartTime', 'EndTime', 'Category'].
        video_dir: Directory containing original videos.
        output_dir: Directory to save generated samples.
        output_csv: Path to save the CSV with video info and labels.
    """
    os.makedirs(output_dir, exist_ok=True)
    data = []

    for _, row in df.iterrows():
        video_id = row['VideoID']
        start_time = row['StartTime']
        end_time = row['EndTime']
        duration = end_time - start_time
        original_path = os.path.join(video_dir, f"{video_id}_{start_time}_{end_time}.mp4")

        # Skip if original video doesn't exist
        if not os.path.exists(original_path):
            continue

        # Randomly cut into >=1 pieces, each >1s
        cuts = []
        if duration > 2:  # Only cut if duration allows for >=2 pieces
            num_cuts = random.randint(1, min(3, int(duration) - 1))  # 1-3 cuts
            cut_points = sorted([start_time] + 
                               [round(start_time + random.uniform(1, duration - 1), 2) 
                                for _ in range(num_cuts)] + 
                               [end_time])
            cuts = [(cut_points[i], cut_points[i+1]) for i in range(len(cut_points)-1)]
        else:
            cuts = [(start_time, end_time)]

        for cut_start, cut_end in cuts:
            cut_duration = cut_end - cut_start
            is_good = random.choice([True, False])  # Randomly decide if this piece is good or bad

            if is_good:
                # Save as good sample
                output_path = os.path.join(output_dir, f"{video_id}_{cut_start}_{cut_end}_good.mp4")
                cmd = [
                    "ffmpeg", "-loglevel", "error",
                    "-i", original_path,
                    "-ss", str(cut_start - start_time),  # Adjust for the cut
                    "-t", str(cut_duration),
                    "-c:v", "libx264", "-c:a", "aac", "-y",
                    output_path
                ]
                label = "Positive"
                misalignment_type = "None"
            else:
                # Generate bad sample
                misalignment_type = random.choice(["time_shift", "noise", "mute", "distort"])
                output_path = os.path.join(output_dir, f"{video_id}_{cut_start}_{cut_end}_bad_{misalignment_type}.mp4")

                if misalignment_type == "time_shift":
                    # Time-shift audio by ±0.5s
                    shift = random.uniform(-0.5, 0.5)
                    cmd = [
                        "ffmpeg", "-loglevel", "error",
                        "-i", original_path,
                        "-ss", str(cut_start - start_time),
                        "-t", str(cut_duration),
                        "-c:v", "libx264", "-c:a", "aac",
                        "-filter_complex", f"adelay={int(shift * 1000)}|{int(shift * 1000)}",
                        "-y", output_path
                    ]
                elif misalignment_type == "noise":
                    # Replace audio with white noise
                    temp_audio = os.path.join(output_dir, "temp_audio.wav")
                    # Extract original audio duration
                    cmd_extract = [
                        "ffmpeg", "-loglevel", "error",
                        "-i", original_path,
                        "-ss", str(cut_start - start_time),
                        "-t", str(cut_duration),
                        "-vn", "-acodec", "pcm_s16le", "-y", temp_audio
                    ]
                    subprocess.run(cmd_extract, check=True)
                    # Generate noise
                    audio = AudioSegment.from_wav(temp_audio)
                    noise = WhiteNoise().to_audio_segment(duration=len(audio))
                    noise.export(temp_audio, format="wav")
                    # Merge noise with video
                    cmd = [
                        "ffmpeg", "-loglevel", "error",
                        "-i", original_path,
                        "-ss", str(cut_start - start_time),
                        "-t", str(cut_duration),
                        "-i", temp_audio,
                        "-c:v", "libx264", "-map", "0:v:0", "-map", "1:a:0",
                        "-y", output_path
                    ]
                    os.remove(temp_audio)
                elif misalignment_type == "mute":
                    # Remove audio
                    cmd = [
                        "ffmpeg", "-loglevel", "error",
                        "-i", original_path,
                        "-ss", str(cut_start - start_time),
                        "-t", str(cut_duration),
                        "-c:v", "libx264", "-an", "-y",
                        output_path
                    ]
                elif misalignment_type == "distort":
                    # Distort video (blur + frame drops)
                    cmd = [
                        "ffmpeg", "-loglevel", "error",
                        "-i", original_path,
                        "-ss", str(cut_start - start_time),
                        "-t", str(cut_duration),
                        "-c:v", "libx264", "-vf", "boxblur=2:1",
                        "-r", "10",  # Reduce frame rate to 10fps
                        "-c:a", "aac", "-y",
                        output_path
                    ]
                label = "Negative"

            try:
                subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
                data.append({
                    "VideoID": video_id,
                    "StartTime": cut_start,
                    "EndTime": cut_end,
                    "Duration": cut_duration,
                    "Label": label,
                    "MisalignmentType": misalignment_type,
                    "Category": row['Category'],
                    "FilePath": output_path
                })
            except subprocess.CalledProcessError as e:
                print(f"Failed to generate sample: {e.stderr.decode().strip()}")

    # Save metadata to CSV
    pd.DataFrame(data).to_csv(output_csv, index=False)
    print(f"Generated samples saved to {output_dir}. Metadata saved to {output_csv}.")
