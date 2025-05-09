import os
import random
import pandas as pd
import subprocess

def generate_samples(df: pd.DataFrame, video_dir: str, output_dir: str, output_csv: str, seed: int = 42) -> None:
    """
    Generate synthetic samples with various audio-visual misalignments from the original videos.
    
    This function processes each video segment and creates both positive (aligned) and negative (misaligned) samples.
    For each segment, it:
    1. Randomly decides whether to create a positive or negative sample
    2. For positive samples: extracts the segment as-is
    3. For negative samples: applies one of four types of misalignments:
       - Time shift: Delays audio relative to video
       - Noise: Adds white noise to audio
       - Mute: Removes audio track
       - Distort: Applies waveform distortion to audio
    
    Args:
        df (pd.DataFrame): DataFrame containing video annotations with columns:
            - VideoID: Unique identifier for each video
            - StartTime: Start time of the segment in seconds
            - EndTime: End time of the segment in seconds
            - Category: Category of the audio-visual event
        video_dir (str): Directory containing the original video files
        output_dir (str): Directory to save the generated samples
        output_csv (str): Path to save the metadata CSV file
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    data = []

    for _, row in df.iterrows():
        video_id = row['VideoID']
        start_time = float(row['StartTime'])
        end_time = float(row['EndTime'])
        duration = end_time - start_time
        original_path = os.path.join(video_dir, f"{video_id}.mp4")

        if not os.path.exists(original_path):
            print(f"✗ Missing file: {original_path}")
            continue

        # Minimum segment length in seconds
        min_seg_len = 1.0
        cuts = []

        # # If segment is long enough, split it into multiple sub-segments
        # if duration >= 2 * min_seg_len:
        #     max_cuts = int(duration // min_seg_len) - 1
        #     num_cuts = random.randint(1, min(3, max_cuts))
        #     cut_points = [start_time]

        #     # Generate random cut points ensuring minimum segment length
        #     for _ in range(num_cuts):
        #         last = cut_points[-1]
        #         remaining_needed = (num_cuts - len(cut_points) + 1) * min_seg_len
        #         max_cut = end_time - remaining_needed
        #         if last + min_seg_len > max_cut:
        #             break
        #         next_cut = round(random.uniform(last + min_seg_len, max_cut), 2)
        #         cut_points.append(next_cut)

        #     cut_points.append(end_time)
        #     cuts = [(round(cut_points[i], 2), round(cut_points[i + 1], 2))
        #             for i in range(len(cut_points) - 1)]
        # else:
        #     cuts = [(round(start_time, 2), round(end_time, 2))]

        if duration >= 2 * min_seg_len:
            cuts = []
            remaining = duration
            current = start_time

            while remaining >= min_seg_len:
                # Sample duration between 2s and 6s or remaining
                seg_len = round(random.uniform(2.0, min(6.0, remaining)), 2)
                cut_end = current + seg_len
                if cut_end > end_time:
                    break
                cuts.append((round(current, 2), round(cut_end, 2)))
                current = cut_end
                remaining = end_time - current
        else:
            cuts = [(round(start_time, 2), round(end_time, 2))]

        # Process each sub-segment
        for cut_start, cut_end in cuts:
            cut_duration = round(cut_end - cut_start, 2)
            if cut_duration < min_seg_len:
                print(f"✗ Skipped (too short < 1s): {video_id} from {cut_start}s to {cut_end}s ({cut_duration:.2f}s)")
                continue

            # Randomly decide whether to create a positive or negative sample
            is_good = random.choice([True, False])
            cut_start_fmt = f"{cut_start:.2f}"
            cut_end_fmt = f"{cut_end:.2f}"

            if is_good:
                # Generate positive sample (unaltered segment)
                output_path = os.path.join(output_dir, f"{video_id}_{cut_start_fmt}_{cut_end_fmt}_good.mp4")
                cmd = [
                    "ffmpeg", "-loglevel", "error",
                    "-i", original_path,
                    "-ss", str(cut_start),
                    "-t", str(cut_duration),
                    "-c:v", "libx264", "-c:a", "aac", "-y",
                    output_path
                ]
                metadata = {
                    "Label": "Positive",
                    "MisalignmentType": "None"
                }
            else:
                # Generate negative sample with random misalignment
                misalignment = random.choice(["time_shift", "noise", "mute", "distort"])
                if misalignment == "time_shift" and cut_start == 0.0:
                    misalignment = random.choice(["noise", "mute", "distort"])

                # Check audio duration and segment length for time shift
                if misalignment == "time_shift":
                    probe_cmd = [
                        "ffprobe", "-v", "error", "-select_streams", "a",
                        "-show_entries", "stream=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                        original_path
                    ]
                    result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    try:
                        audio_duration = float(result.stdout.decode().strip())
                    except:
                        audio_duration = 0.0

                    # Ensure safe time shift (max 80% of segment duration)
                    max_safe_shift = int(cut_duration * 1000 * 0.8)
                    if audio_duration < cut_duration + 0.5 or max_safe_shift < 100:
                        print(f"✗ Skipped time_shift: unsafe audio or segment too short ({cut_duration:.2f}s)")
                        misalignment = random.choice(["noise", "mute", "distort"])

                output_path = os.path.join(output_dir,
                                           f"{video_id}_{cut_start_fmt}_{cut_end_fmt}_bad_{misalignment}.mp4")

                # Apply selected misalignment
                if misalignment == "time_shift":
                    # Add random delay to audio (100ms to 80% of segment duration)
                    shift_ms = random.randint(100, max(100, int(cut_duration * 1000 * 0.8)))
                    cmd = [
                        "ffmpeg", "-loglevel", "error",
                        "-i", original_path,
                        "-ss", str(cut_start),
                        "-t", str(cut_duration),
                        "-filter_complex", f"adelay={shift_ms}|{shift_ms}",
                        "-c:v", "copy",
                        "-y", output_path
                    ]

                elif misalignment == "noise":
                    # Add white noise to audio track
                    cmd = [
                        "ffmpeg", "-loglevel", "error",
                        "-i", original_path,
                        "-ss", str(cut_start),
                        "-t", str(cut_duration),
                        "-f", "lavfi", "-i", "anoisesrc=color=white",
                        "-c:v", "libx264", "-map", "0:v", "-map", "1:a",
                        "-shortest", "-y", output_path
                    ]

                elif misalignment == "mute":
                    # Remove audio track completely
                    cmd = [
                        "ffmpeg", "-loglevel", "error",
                        "-i", original_path,
                        "-ss", str(cut_start),
                        "-t", str(cut_duration),
                        "-an",
                        "-c:v", "copy",
                        "-y", output_path
                    ]

                elif misalignment == "distort":
                    # Apply waveform distortion to audio
                    cmd = [
                        "ffmpeg", "-loglevel", "error",
                        "-i", original_path,
                        "-ss", str(cut_start),
                        "-t", str(cut_duration),
                        "-filter_complex", "setpts='0.75*PTS+sin(N*0.05)*0.05/TB'",
                        "-c:a", "copy",
                        "-y", output_path
                    ]

                metadata = {
                    "Label": "Negative",
                    "MisalignmentType": misalignment
                }

            # Process the video segment
            try:
                subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                data.append({
                    "VideoID": video_id,
                    "StartTime": cut_start,
                    "EndTime": cut_end,
                    **metadata,
                    "Category": row['Category'],
                    "FilePath": output_path
                })
                print(f"✓ Generated: {output_path}")
            except subprocess.CalledProcessError as e:
                print("✗ ffmpeg FAILED")
                print(f"  VideoID: {video_id}")
                print(f"  Segment: {cut_start}s to {cut_end}s (Duration: {cut_duration:.2f}s)")
                print(f"  Cmd: {' '.join(cmd)}")
                print(f"  Error: {e.stderr.decode(errors='ignore').strip()}")

    # Save metadata to CSV
    pd.DataFrame(data).to_csv(output_csv, index=False)
    print(f"\n✓ Saved metadata to {output_csv}")
