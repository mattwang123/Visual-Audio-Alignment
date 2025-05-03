import os
import random
import pandas as pd
import subprocess

def generate_samples(df: pd.DataFrame, video_dir: str, output_dir: str, output_csv: str, seed: int = 42) -> None:
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

        min_seg_len = 1.0
        cuts = []

        if duration >= 2 * min_seg_len:
            max_cuts = int(duration // min_seg_len) - 1
            num_cuts = random.randint(1, min(3, max_cuts))
            cut_points = [start_time]

            for _ in range(num_cuts):
                last = cut_points[-1]
                remaining_needed = (num_cuts - len(cut_points) + 1) * min_seg_len
                max_cut = end_time - remaining_needed
                if last + min_seg_len > max_cut:
                    break
                next_cut = round(random.uniform(last + min_seg_len, max_cut), 2)
                cut_points.append(next_cut)

            cut_points.append(end_time)
            cuts = [(round(cut_points[i], 2), round(cut_points[i + 1], 2))
                    for i in range(len(cut_points) - 1)]
        else:
            cuts = [(round(start_time, 2), round(end_time, 2))]

        for cut_start, cut_end in cuts:
            cut_duration = round(cut_end - cut_start, 2)
            if cut_duration < min_seg_len:
                print(f"✗ Skipped (too short < 1s): {video_id} from {cut_start}s to {cut_end}s ({cut_duration:.2f}s)")
                continue

            is_good = random.choice([True, False])
            cut_start_fmt = f"{cut_start:.2f}"
            cut_end_fmt = f"{cut_end:.2f}"

            if is_good:
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
                misalignment = random.choice(["time_shift", "noise", "mute", "distort"])
                if misalignment == "time_shift" and cut_start == 0.0:
                    misalignment = random.choice(["noise", "mute", "distort"])

                # Check and possibly fallback if audio is missing or segment is too short for shift
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

                    max_safe_shift = int(cut_duration * 1000 * 0.8)
                    if audio_duration < cut_duration + 0.5 or max_safe_shift < 100:
                        print(f"✗ Skipped time_shift: unsafe audio or segment too short ({cut_duration:.2f}s)")
                        misalignment = random.choice(["noise", "mute", "distort"])

                output_path = os.path.join(output_dir,
                                           f"{video_id}_{cut_start_fmt}_{cut_end_fmt}_bad_{misalignment}.mp4")

                # Final cmd setup after confirming actual misalignment
                if misalignment == "time_shift":
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
                    speed_factor = round(random.uniform(0.85, 1.15), 2)
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

    pd.DataFrame(data).to_csv(output_csv, index=False)
    print(f"\n✓ Saved metadata to {output_csv}")
