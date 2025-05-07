import numpy as np
import sys
import os
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import utility functions
from utils.evaluation_utils import evaluate_predictions, plot_confusion_matrix, plot_roc_curve
from utils.dataloader_utils import unified_split, load_combined_npz_features


def detect_peaks(signal, threshold_factor=0.6, min_distance=5):
    """
    Detect peaks in a signal based on a dynamic threshold.
    
    Args:
        signal: Input signal array
        threshold_factor: Factor of signal max used as threshold
        min_distance: Minimum samples between peaks
        
    Returns:
        Array of indices where peaks occur
    """
    if len(signal) <= 1:
        return np.array([])
    
    # Normalize signal to [0,1]
    if np.ptp(signal) > 0:
        normalized = (signal - np.min(signal)) / np.ptp(signal)
    else:
        return np.array([])
    
    # Dynamic threshold based on signal characteristics
    threshold = threshold_factor * np.max(normalized)
    
    # Find all points above threshold
    above_threshold = np.where(normalized >= threshold)[0]
    
    if len(above_threshold) == 0:
        return np.array([])
    
    # Extract peaks with minimum distance constraint
    peaks = [above_threshold[0]]
    
    for i in above_threshold:
        if i - peaks[-1] >= min_distance:
            peaks.append(i)
    
    return np.array(peaks)


def compute_alignment_score(audio_feat, visual_feat):
    """
    Compute alignment score between audio and visual features based on relative peak synchronization.
    Lower score means better alignment (less anomalous).
    
    This implementation focuses on temporal alignment of relative peaks in each modality,
    regardless of absolute intensity levels.
    """
    if not isinstance(audio_feat, np.ndarray) or not isinstance(visual_feat, np.ndarray):
        return 1.0
    
    # Flatten if multidimensional
    if audio_feat.ndim > 1:
        audio_feat = audio_feat.flatten()
    if visual_feat.ndim > 1:
        visual_feat = visual_feat.flatten()
    
    # Ensure same length
    min_len = min(len(audio_feat), len(visual_feat))
    if min_len <= 10:  # Need reasonable length for peak detection
        return 1.0
    
    a = audio_feat[:min_len]
    v = visual_feat[:min_len]
    
    # Normalize to [0,1]
    if np.ptp(a) > 0:
        a = (a - np.min(a)) / np.ptp(a)
    else:
        a = np.zeros_like(a)
        
    if np.ptp(v) > 0:
        v = (v - np.min(v)) / np.ptp(v)
    else:
        v = np.zeros_like(v)
    
    # Detect relative peaks in each modality
    # Use a lower threshold to capture more relative peaks
    audio_peaks = detect_peaks(a, threshold_factor=0.5, min_distance=3)
    visual_peaks = detect_peaks(v, threshold_factor=0.5, min_distance=3)
    
    # If either modality has no peaks, it's likely an anomaly
    if len(audio_peaks) == 0 or len(visual_peaks) == 0:
        return 0.9  # High anomaly score but not maximum
    
    # Create activity window around each peak
    window_size = min(10, min_len // 10)  # Adaptive window size
    
    # Calculate peak synchronization score
    synchronization_scores = []
    
    # For each audio peak, find the closest visual peak
    for ap in audio_peaks:
        if len(visual_peaks) == 0:
            synchronization_scores.append(1.0)  # Maximum distance
            continue
            
        # Find closest visual peak
        distances = np.abs(visual_peaks - ap)
        min_distance = np.min(distances)
        min_idx = np.argmin(distances)
        
        # Normalize distance by the window size
        normalized_distance = min(min_distance / window_size, 1.0)
        
        # Check if peaks are within reasonable temporal window
        if min_distance <= window_size:
            # If within window, check consistency of peak response
            audio_peak_height = a[ap]
            visual_peak_height = v[visual_peaks[min_idx]]
            
            # Peaks should be proportional in properly synchronized content
            # Calculate height similarity (1 = same height, 0 = very different heights)
            height_similarity = 1.0 - min(abs(audio_peak_height - visual_peak_height), 1.0)
            
            # Combine temporal distance and peak height similarity
            peak_score = 0.7 * normalized_distance + 0.3 * (1.0 - height_similarity)
            synchronization_scores.append(peak_score)
        else:
            # Peaks too far apart
            synchronization_scores.append(1.0)
    
    # Calculate peak density similarity
    # Similar number of peaks should occur in both streams in normal content
    peak_density_ratio = abs(len(audio_peaks) - len(visual_peaks)) / max(len(audio_peaks), len(visual_peaks))
    
    # Calculate proportion of matched peaks
    matched_peaks_ratio = min(1.0, sum(1 for s in synchronization_scores if s < 0.5) / len(synchronization_scores))
    proportion_unmatched = 1.0 - matched_peaks_ratio
    
    # Synchronization anomaly score (higher = more anomalous)
    if synchronization_scores:
        mean_sync_score = np.mean(synchronization_scores)
    else:
        mean_sync_score = 1.0
    
    # Check for temporal consistency in synchronization
    temporal_consistency = 1.0
    if len(synchronization_scores) >= 3:
        # Measure variability in synchronization quality over time
        sync_variability = np.std(synchronization_scores) / max(0.1, np.mean(synchronization_scores))
        temporal_consistency = min(1.0, sync_variability)
    
    # Final anomaly score combining multiple factors
    final_score = (
        0.5 * mean_sync_score +           # Average peak synchronization quality
        0.3 * proportion_unmatched +      # Proportion of unmatched peaks
        0.1 * peak_density_ratio +        # Similarity of peak distribution
        0.1 * temporal_consistency        # Consistency of synchronization over time
    )
    
    return final_score


def threshold_evaluation(train_scores, train_labels, test_scores, test_labels, config=None, save_output=True):
    valid_train = [i for i, l in enumerate(train_labels) if l is not None]
    train_scores_clean = train_scores[valid_train]
    train_labels_clean = train_labels[valid_train]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("thresholding_output", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    roc_path = os.path.join(output_dir, "roc_curve.png")
    threshold = plot_roc_curve(train_labels_clean, train_scores_clean, out_path=roc_path)
    print(f"✅ Selected threshold: {threshold:.3f}")
    
    metrics = evaluate_predictions(test_labels, test_scores, threshold=threshold)
    
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(metrics, out_path=cm_path)
    
    # Save metrics and config
    # Convert all NumPy types in metrics to native Python types
    def to_serializable(d):
        return {k: (v.item() if isinstance(v, (np.generic, np.ndarray)) else v) for k, v in d.items()}
    
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(to_serializable(metrics), f, indent=4)
    
    if config is not None:
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
    
    log_summary = (
        f"Thresholding Summary - {timestamp}\n"
        f"Selected Threshold: {threshold:.3f}\n"
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall: {metrics['recall']:.4f}\n"
        f"F1 Score: {metrics['f1_score']:.4f}\n"
    )
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(log_summary)
    
    print(f"\n📁 All thresholding artifacts saved to: {output_dir}")
    return metrics


# === Main Execution ===
def main():
    # Configuration
    npz_dir = r"D:\lenovo\mia_final_project\preprocessed_output"
    print("🚀 Running threshold-based alignment model with peak analysis")
    
    # Unified Train/Test Split
    train_files, test_files = unified_split(npz_dir, test_size=0.3, seed=42)
    
    # Load and Organize Features
    all_features = load_combined_npz_features(npz_dir)
    
    train_features = {
        os.path.basename(p): all_features[os.path.basename(p)]
        for p in train_files if os.path.basename(p) in all_features
    }
    
    test_features = {
        os.path.basename(p): all_features[os.path.basename(p)]
        for p in test_files if os.path.basename(p) in all_features
    }
    
    print(f"✅ Loaded {len(train_features)} training samples and {len(test_features)} testing samples")
    
    # Compute Alignment Scores and Labels
    print("📊 Computing alignment scores with peak-based algorithm...")
    
    train_scores = np.array([
        compute_alignment_score(f['audio'], f['visual']) for f in train_features.values()
    ])
    train_labels = np.array([f['label'] for f in train_features.values()])
    
    test_scores = np.array([
        compute_alignment_score(f['audio'], f['visual']) for f in test_features.values()
    ])
    test_labels = np.array([f['label'] for f in test_features.values()])
    
    # Configuration details for logging
    config = {
        "model_type": "peak_synchronization_threshold",
        "peak_detection": {
            "threshold_factor": 0.5,
            "min_distance": 3
        },
        "score_weights": {
            "mean_sync_score": 0.5,
            "proportion_unmatched": 0.3,
            "peak_density_ratio": 0.1,
            "temporal_consistency": 0.1
        },
        "train_samples": len(train_features),
        "test_samples": len(test_features),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Run Evaluation
    print("🧪 Evaluating model performance...")
    metrics = threshold_evaluation(train_scores, train_labels, test_scores, test_labels, config=config)
    
    # Print Final Metrics
    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    
    return metrics


if __name__ == "__main__":
    main()