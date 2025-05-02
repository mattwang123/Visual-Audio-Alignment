from IPython.display import display, HTML
import pandas as pd
import os
import subprocess

def print_color(message, color):
    """Helper function to display colored text in Jupyter"""
    display(HTML(f'<text style="color:{color}">{message}</text>'))

def trim_videos(df: pd.DataFrame, video_dir: str, output_dir: str) -> None:
    """
    Trim videos to the annotated segments using ffmpeg.
    
    Args:
        df: DataFrame with columns ['VideoID', 'StartTime', 'EndTime'].
        video_dir: Directory containing original videos.
        output_dir: Directory to save trimmed clips.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for _, row in df.iterrows():
        video_id = row['VideoID']
        start_time = row['StartTime']
        end_time = row['EndTime']
        duration = end_time - start_time
        
        input_path = os.path.join(video_dir, f"{video_id}.mp4")
        output_path = os.path.join(output_dir, f"{video_id}_{start_time}_{end_time}.mp4")
        
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-i", input_path,
            "-ss", str(start_time),
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-y",
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
            print_color(f"✓ Successfully trimmed: {output_path}", "green")
        except subprocess.CalledProcessError as e:
            print_color(f"✗ Failed to trim {video_id}: {e.stderr.decode().strip()}", "red")
        except KeyboardInterrupt:
            print_color("⚠️ Video trimming interrupted by user", "orange")
            raise