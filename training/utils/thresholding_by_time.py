import numpy as np
import os
import json
from datetime import datetime
from utils.evaluation_utils import evaluate_predictions, plot_confusion_matrix, plot_roc_curve


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
    Compute alignment score between audio and visual features based on peak analysis.
    Lower score means better alignment (less anomalous).
    
    This implementation considers:
    1. Temporal alignment of peaks between audio and visual features
    2. Expected complementary pattern of peaks
    3. Relationship between overall energy distributions
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
    
    # 1. Peak analysis
    audio_peaks = detect_peaks(a)
    visual_peaks = detect_peaks(v)
    
    # If no peaks detected in either stream, likely an anomaly
    if len(audio_peaks) == 0 and len(visual_peaks) == 0:
        return 1.0
    
    # 2. Peak timing relationship analysis
    peak_alignment_score = 0.5  # Default middle value
    
    if len(audio_peaks) > 0 and len(visual_peaks) > 0:
        # Create peak activity timeline
        audio_timeline = np.zeros(min_len)
        visual_timeline = np.zeros(min_len)
        
        # Add gaussian bumps around peaks to represent activity windows
        for peak in audio_peaks:
            window = 5  # Window around each peak
            start = max(0, peak - window)
            end = min(min_len, peak + window + 1)
            audio_timeline[start:end] = 1
            
        for peak in visual_peaks:
            window = 5  # Window around each peak
            start = max(0, peak - window)
            end = min(min_len, peak + window + 1)
            visual_timeline[start:end] = 1
        
        # Calculate timing relationships
        # 1. Causal relationship: visual peaks often follow audio peaks
        # 2. Inverse relationship: high audio during low visual and vice versa
        
        # Measure causal relationship (normal: visual peaks slightly after audio peaks)
        shifted_visual = np.zeros_like(visual_timeline)
        shift_amount = 2  # Visual peaks typically follow audio peaks by a small delay
        shifted_visual[shift_amount:] = visual_timeline[:-shift_amount]
        causal_match = np.mean(np.logical_and(audio_timeline, shifted_visual))
        
        # Measure inverse relationship (complementary pattern)
        # When audio is high, visual should be lower and vice versa
        inverse_match = np.mean(np.logical_xor(audio_timeline, visual_timeline))
        
        # Combine these relationship scores
        peak_alignment_score = 1.0 - (0.6 * causal_match + 0.4 * inverse_match)
    
    # 3. Calculate overall energy correlation and distribution
    correlation = np.corrcoef(a, v)[0, 1] if min_len > 1 else 0
    if np.isnan(correlation):
        correlation = 0
    
    # Strong negative correlation is expected (inverse relationship)
    correlation_score = min(1.0 - abs(correlation), 1.0 - max(0, correlation))
    
    # Analyze pattern consistency
    pattern_score = np.mean(np.abs(a + v - 1.0))
    
    # Detect anomalous patterns
    both_high_ratio = np.mean(np.logical_and(a > 0.7, v > 0.7))
    both_low_ratio = np.mean(np.logical_and(a < 0.2, v < 0.2))
    anomaly_pattern = 0.5 * (both_high_ratio + both_low_ratio)
    
    # Combine all scores with weights
    final_score = (
        0.5 * peak_alignment_score +   # Peak timing analysis
        0.2 * correlation_score +      # Overall correlation
        0.2 * pattern_score +          # Complementary pattern
        0.1 * anomaly_pattern          # Anomalous joint patterns
    )
    
    return final_score


def compute_all_alignment_scores(features_dict):
    scores, filenames = [], []
    for fname, data in features_dict.items():
        try:
            score = compute_alignment_score(data['audio'], data['visual'])
            scores.append(score)
            filenames.append(fname)
        except Exception as e:
            print(f"❌ Error computing alignment for {fname}: {e}")
    if not scores:
        raise ValueError("No valid pairs found for alignment computation")
    print(f"✅ Computed {len(scores)} alignment scores")
    return np.array(scores), filenames


def extract_labels(features_dict):
    labels = []
    for fname, data in features_dict.items():
        label = data.get("label", None)
        if label in [0, 1]:
            labels.append(label)
        else:
            print(f"⚠️ Skipped invalid label in {fname}")
            labels.append(None)
    return np.array(labels)


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