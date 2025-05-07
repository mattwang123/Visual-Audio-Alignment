import numpy as np
import os
import json
from datetime import datetime
from scipy import signal
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

from utils.evaluation_utils import evaluate_predictions, plot_confusion_matrix, plot_roc_curve

def compute_alignment_score(audio_feat, visual_feat, method='cross_corr'):
    """
    Compute alignment score between audio and visual features using various methods.
    
    Args:
        audio_feat: Audio feature vector (energy of sound)
        visual_feat: Visual feature vector (magnitude of optical flow)
        method: Alignment method to use ('cross_corr', 'dynamic_time_warp', or 'pearson')
    
    Returns:
        Alignment score (lower is better for cross_corr and DTW, higher is better for pearson)
    """
    if not isinstance(audio_feat, np.ndarray) or not isinstance(visual_feat, np.ndarray):
        return 1.0 if method != 'pearson' else 0.0
    
    # Convert to 1D arrays if needed
    audio_feat = audio_feat.flatten()
    visual_feat = visual_feat.flatten()
    
    min_len = min(len(audio_feat), len(visual_feat))
    if min_len == 0:
        return 1.0 if method != 'pearson' else 0.0
    
    # Normalize features using z-score normalization
    scaler = StandardScaler()
    a = scaler.fit_transform(audio_feat[:min_len].reshape(-1, 1)).flatten()
    v = scaler.fit_transform(visual_feat[:min_len].reshape(-1, 1)).flatten()
    
    if method == 'cross_corr':
        # Compute cross-correlation and find peak
        corr = signal.correlate(a, v, mode='valid')
        if len(corr) == 0:
            return 1.0
        # The score is inversely related to the maximum correlation
        return 1.0 - (np.max(corr) / (np.linalg.norm(a) * np.linalg.norm(v)))
    
    elif method == 'dynamic_time_warp':
        # Dynamic Time Warping distance
        # (Implementation simplified - consider using fastdtw package for production)
        dtw_matrix = np.zeros((min_len, min_len))
        dtw_matrix[0, 0] = abs(a[0] - v[0])
        for i in range(1, min_len):
            dtw_matrix[i, 0] = dtw_matrix[i-1, 0] + abs(a[i] - v[0])
        for j in range(1, min_len):
            dtw_matrix[0, j] = dtw_matrix[0, j-1] + abs(a[0] - v[j])
        for i in range(1, min_len):
            for j in range(1, min_len):
                cost = abs(a[i] - v[j])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],   # insertion
                                            dtw_matrix[i, j-1],   # deletion
                                            dtw_matrix[i-1, j-1]) # match
        return dtw_matrix[-1, -1] / min_len  # normalized distance
    
    elif method == 'pearson':
        # Pearson correlation coefficient (higher is better)
        r, _ = pearsonr(a, v)
        return -r  # return negative so lower is better (consistent with other methods)
    
    else:
        raise ValueError(f"Unknown alignment method: {method}")


def compute_all_alignment_scores(features_dict, method='cross_corr'):
    """
    Compute alignment scores for all files in features_dict.
    
    Args:
        features_dict: Dictionary containing audio and visual features
        method: Alignment method to use
    
    Returns:
        scores: Array of alignment scores
        filenames: List of corresponding filenames
    """
    scores, filenames = [], []
    for fname, data in features_dict.items():
        try:
            score = compute_alignment_score(data['audio'], data['visual'], method=method)
            scores.append(score)
            filenames.append(fname)
        except Exception as e:
            print(f"❌ Error computing alignment for {fname}: {e}")
            scores.append(1.0 if method != 'pearson' else 0.0)
            filenames.append(fname)
    
    if not scores:
        raise ValueError("No valid pairs found for alignment computation")
    
    print(f"✅ Computed {len(scores)} alignment scores using {method} method")
    return np.array(scores), filenames


def extract_labels(features_dict):
    """Extract labels from features dictionary, handling invalid labels."""
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
    """
    Evaluate performance using optimal threshold and save results.
    
    Args:
        train_scores: Alignment scores for training set
        train_labels: Labels for training set
        test_scores: Alignment scores for test set
        test_labels: Labels for test set
        config: Optional configuration dictionary to save
        save_output: Whether to save output files
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Filter out None labels
    valid_train = [i for i, l in enumerate(train_labels) if l is not None]
    train_scores_clean = train_scores[valid_train]
    train_labels_clean = train_labels[valid_train]
    
    valid_test = [i for i, l in enumerate(test_labels) if l is not None]
    test_scores_clean = test_scores[valid_test]
    test_labels_clean = test_labels[valid_test]
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join("thresholding_output", f"run_{timestamp}")
    
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find optimal threshold and evaluate
    roc_path = os.path.join(output_dir, "roc_curve.png") if save_output else None
    threshold = plot_roc_curve(train_labels_clean, train_scores_clean, out_path=roc_path)
    print(f"✅ Selected threshold: {threshold:.3f}")
    
    metrics = evaluate_predictions(test_labels_clean, test_scores_clean, threshold=threshold)
    
    if save_output:
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix(metrics, out_path=cm_path)
        
        # Save metrics and config
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