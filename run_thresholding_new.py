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


def detect_peaks(signal, threshold_factor=0.4, min_distance=3):
    """
    Detect peaks in a signal using a more robust approach with adaptive thresholding.
    
    Args:
        signal: Input signal array
        threshold_factor: Factor of signal range used as threshold
        min_distance: Minimum samples between peaks
        
    Returns:
        Array of indices where peaks occur
    """
    if len(signal) <= 3:
        return np.array([])
    
    # Normalize signal to [0,1]
    if np.ptp(signal) > 0:
        normalized = (signal - np.min(signal)) / np.ptp(signal)
    else:
        return np.array([])
    
    # Use a moving average to establish a dynamic baseline
    window_size = max(3, min(15, len(normalized) // 10))
    padded = np.pad(normalized, (window_size//2, window_size//2), mode='edge')
    moving_avg = np.zeros_like(normalized)
    
    for i in range(len(normalized)):
        moving_avg[i] = np.mean(padded[i:i+window_size])
    
    # Find local peaks above the moving average
    peaks = []
    for i in range(1, len(normalized)-1):
        # Check if point is a local maximum
        if normalized[i] > normalized[i-1] and normalized[i] >= normalized[i+1]:
            # Check if it's significantly above the local baseline
            if normalized[i] > moving_avg[i] + threshold_factor * np.ptp(normalized):
                # If we already have peaks, check the minimum distance
                if not peaks or i - peaks[-1] >= min_distance:
                    peaks.append(i)
    
    return np.array(peaks)


def compute_alignment_score(audio_feat, visual_feat):
    """
    Compute alignment score between audio and visual features.
    Lower score means better alignment (less anomalous).
    
    This refined implementation uses multiple techniques to detect anomalies:
    1. Cross-correlation to identify temporal offset
    2. Peak alignment with allowed delay window
    3. Signal envelope similarity
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
    if min_len <= 10:  # Need reasonable length for analysis
        return 1.0
    
    a = audio_feat[:min_len].astype(np.float64)
    v = visual_feat[:min_len].astype(np.float64)
    
    # Normalize to [0,1]
    if np.ptp(a) > 0:
        a = (a - np.min(a)) / np.ptp(a)
    else:
        a = np.zeros_like(a)
        
    if np.ptp(v) > 0:
        v = (v - np.min(v)) / np.ptp(v)
    else:
        v = np.zeros_like(v)
    
    # 1. Cross-correlation to find optimal temporal alignment
    max_lag = min(min_len // 5, 30)  # Maximum allowed lag in either direction
    if min_len > (2 * max_lag + 1):
        # Calculate cross-correlation for lags from -max_lag to +max_lag
        corrs = np.zeros(2 * max_lag + 1)
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corrs[lag + max_lag] = np.corrcoef(a[-lag:], v[:min_len+lag])[0, 1]
            elif lag > 0:
                corrs[lag + max_lag] = np.corrcoef(a[:min_len-lag], v[lag:])[0, 1]
            else:
                corrs[lag + max_lag] = np.corrcoef(a, v)[0, 1]
        
        # Replace NaN values with 0
        corrs = np.nan_to_num(corrs)
        
        # Best correlation coefficient and corresponding lag
        best_corr = np.max(corrs)
        optimal_lag = np.argmax(corrs) - max_lag
        
        # Score based on correlation strength and lag size
        correlation_score = 1.0 - abs(best_corr)
        lag_factor = min(1.0, abs(optimal_lag) / max_lag)
        
        # Apply the optimal lag for subsequent analysis
        if optimal_lag < 0:
            a_aligned = a[-optimal_lag:]
            v_aligned = v[:min_len+optimal_lag]
        elif optimal_lag > 0:
            a_aligned = a[:min_len-optimal_lag]
            v_aligned = v[optimal_lag:]
        else:
            a_aligned = a
            v_aligned = v
            
        aligned_len = len(a_aligned)
    else:
        # Signal too short for meaningful cross-correlation
        correlation_score = 0.5
        lag_factor = 0.5
        a_aligned = a
        v_aligned = v
        aligned_len = min_len
    
    # 2. Peak analysis on aligned signals
    a_peaks = detect_peaks(a_aligned, threshold_factor=0.3)
    v_peaks = detect_peaks(v_aligned, threshold_factor=0.3)
    
    # Calculate timing match between peaks with tolerance window
    peak_match_score = 0.5  # Default middle value
    
    if len(a_peaks) > 0 and len(v_peaks) > 0:
        tolerance_window = max(3, aligned_len // 50)  # Adaptive tolerance window
        
        # Count matching peaks
        matching_peaks = 0
        for ap in a_peaks:
            # Find closest visual peak
            distances = np.abs(v_peaks - ap)
            if len(distances) > 0 and np.min(distances) <= tolerance_window:
                matching_peaks += 1
        
        # Calculate match ratio based on total audio peaks
        if len(a_peaks) > 0:
            peak_match_ratio = matching_peaks / len(a_peaks)
            peak_match_score = 1.0 - peak_match_ratio
        
        # Penalize if peak counts differ significantly 
        peak_count_ratio = min(len(a_peaks), len(v_peaks)) / max(1, max(len(a_peaks), len(v_peaks)))
        peak_count_penalty = 1.0 - peak_count_ratio
    elif len(a_peaks) == 0 and len(v_peaks) == 0:
        # No peaks in either signal - check if they're both flat
        if np.std(a_aligned) < 0.05 and np.std(v_aligned) < 0.05:
            peak_match_score = 0.0  # Both signals flat, could be normal
        else:
            peak_match_score = 1.0  # Different patterns with no clear peaks
        peak_count_penalty = 0.0
    else:
        # Peaks in only one signal
        peak_match_score = 1.0
        peak_count_penalty = 1.0
    
    # 3. Signal envelope analysis
    # Calculate energy envelopes using moving average
    window_size = max(3, aligned_len // 20)
    
    # Function to get envelope
    def get_envelope(signal, win_size):
        if len(signal) <= win_size:
            return signal
        padded = np.pad(signal, (win_size//2, win_size//2), mode='edge')
        envelope = np.zeros_like(signal)
        for i in range(len(signal)):
            envelope[i] = np.mean(padded[i:i+win_size])
        return envelope
    
    a_envelope = get_envelope(a_aligned, window_size)
    v_envelope = get_envelope(v_aligned, window_size)
    
    # Measure envelope similarity 
    envelope_corr = np.corrcoef(a_envelope, v_envelope)[0, 1]
    if np.isnan(envelope_corr):
        envelope_corr = 0
    
    # Higher correlation (positive or negative) is better
    envelope_score = 1.0 - abs(envelope_corr)
    
    # Combine all scores
    final_score = (
        0.30 * correlation_score +    # Cross-correlation strength
        0.15 * lag_factor +           # Optimal lag amount
        0.30 * peak_match_score +     # Peak alignment
        0.15 * peak_count_penalty +   # Peak count similarity
        0.10 * envelope_score         # Envelope similarity
    )
    
    # Safety check - ensure reasonable value
    final_score = max(0.0, min(1.0, final_score))
    
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
        "model_type": "optimized_cross_correlation_peak_analysis",
        "peak_detection": {
            "threshold_factor": 0.3,
            "min_distance": 3,
            "moving_average_window": "adaptive"
        },
        "score_weights": {
            "correlation_score": 0.30,
            "lag_factor": 0.15,
            "peak_match_score": 0.30,
            "peak_count_penalty": 0.15,
            "envelope_score": 0.10
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