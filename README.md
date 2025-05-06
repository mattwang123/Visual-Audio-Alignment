# Cross-Modal Video Anomaly Detection

This is the final project for 553.493 MIA.

Title: Cross-Modal Video Anomaly Detection: Visual-Audio Alignment via Optical Flow and Fourier Analysis


## Visual-Audio Alignment via Optical Flow and Fourier Analysis

This repository contains the implementation for a cross-modal video anomaly detection system that focuses on identifying audio-visual misalignments in video content. The project leverages both visual and audio features to detect anomalies in the synchronization between audio and visual components.

## Project Overview

Video anomaly detection is a challenging task with applications in security, content moderation, and quality assurance. This project specifically focuses on the alignment between audio and visual elements in videos, identifying cases where these modalities are out of sync or manipulated.

Our approach combines:
- Visual feature extraction using optical flow and spatial information
- Audio feature extraction through MFCC (Mel-Frequency Cepstral Coefficients)
- Multiple detection methods, including simple thresholding, and MLP classification models

## Dataset

This project uses the [AVE Dataset](https://sites.google.com/view/audiovisualresearch), which includes:
- 4,143 videos covering 28 audio-visual event categories
- 10-second video clips with various audio-visual events
- Labels indicating synchronized (1) or non-synchronized (0) segments

The project includes tools to:
- Process the original AVE dataset
- Extract relevant features
- Generate synthetic anomalies for training and evaluation

## Installation
// TODO: zip files?

1. Clone the repository:
```bash
git clone https://github.com/your-username/Visual-Audio-Alignment.git
cd Visual-Audio-Alignment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```


## Usage

### Data Preprocessing
> Relevant codes in `preprocessing` directory.

1. Convert annotations from AVE dataset to `csv` format
   - Purpose: Makes the annotations more accessible and easier to work with in Python

2. Trim videos according to annotations
   - Purpose: Extracts the relevant segments from full videos based on annotation timestamps
     - Output saved in `trimmed_clips` zip file

3. Generate synthetic samples with various misalignments

    In order to create a balanced dataset with both positive (aligned) and negative (misaligned) samples, we utilized original videos and preprocessed annotations to generate synthetic samples. The process involves:

    **Sample Generation Process**:
    1. For each video segment:
       - Randomly decide whether to create a positive or negative sample
       - If segment duration ≥ 2 seconds, split into multiple sub-segments (1-3 cuts)
       - Ensure each sub-segment is at least 1 second long
    
    2. For positive samples:
       - Extract the segment as-is using ffmpeg
       - Preserve both video and audio streams
       - Label as "Positive" with "None" misalignment type
    
    3. For negative samples, apply one of four misalignments:
       - Time shift: 
         * Add random audio delay (100ms to 80% of segment duration)
         * Skip if segment is too short or audio duration is insufficient
       - Noise:
         * Add white noise to audio track while preserving video
         * Use ffmpeg's anoisesrc filter with white noise
       - Mute:
         * Completely remove audio track
         * Keep video stream unchanged
       - Distort:
         * Apply waveform distortion to audio
         * Use ffmpeg's setpts filter for temporal distortion
    
    **Output**:
    - Generated video samples in `generated_samples` directory
    - Metadata in `generated_samples_metadata.csv` containing:
      * VideoID, StartTime, EndTime
      * Label (Positive/Negative)
      * MisalignmentType (None/time_shift/noise/mute/distort)
      * Category and FilePath



### Feature Extraction

Extract visual and audio features from the preprocessed video files:
- Extract visual features (resized grayscale frames) at 5 FPS
- Extract audio features (MFCC coefficients) 
- Save the features as compressed NPZ files

### Model Training and Evaluation

Three different approaches are implemented:

#### Method 1: Simple Thresholding

**Idea**:
- Compare the audio and visual energy curves. If the difference is too large, we infer desynchronization.

**Mathematical Method**:

\(\text{sync\_score}(t) = |V(t) - A(t)|\)

- Set a threshold \(\theta\), and if \(\text{sync\_score}(t) < \theta\), classify as synchronized.
 
**Hyperparameter**:
- \(\theta\) can be optimized using a validation set.

---

#### Method 2: MLP Classifier

**Input**:
- Concatenate visual features \(V(t)\) and audio features \(A(t)\).

**Network Architecture**:
- `Linear(2→32)` → ReLU → `Linear(32→16)` → ReLU → `Linear(16→2)` → Softmax

**Loss Function**:
- Cross-entropy loss

**Training**:
- The input features \([V(t), A(t)]\) are labeled as `1` or `0`.


## Results

The models are evaluated based on:
- Accuracy
- Precision
- Recall
- F1 Score

The simple thresholding approach provides a baseline, while the MLP classifier and transformer-based methods offer improved performance for detecting more subtle anomalies.

