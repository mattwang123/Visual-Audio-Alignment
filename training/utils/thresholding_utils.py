import numpy as np
import os
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score

from utils.evaluation_utils import evaluate_predictions, plot_confusion_matrix, plot_roc_curve


def compute_alignment_score(audio_feat, visual_feat):
    if isinstance(audio_feat, np.ndarray) and isinstance(visual_feat, np.ndarray):
        if audio_feat.ndim > 1:
            audio_feat = audio_feat.flatten()
        if visual_feat.ndim > 1:
            visual_feat = visual_feat.flatten()
        min_len = min(len(audio_feat), len(visual_feat))
        if min_len > 0:
            a = audio_feat[:min_len]
            v = visual_feat[:min_len]
            if np.ptp(a) > 0:
                a = (a - np.min(a)) / np.ptp(a)
            if np.ptp(v) > 0:
                v = (v - np.min(v)) / np.ptp(v)
            return np.linalg.norm(a - v) / np.sqrt(min_len)
    return 1.0


def compute_all_alignment_scores(features_dict):
    scores, filenames = [], []
    for fname, data in features_dict.items():
        try:
            score = compute_alignment_score(data['audio'], data['visual'])
            scores.append(score)
            filenames.append(fname)
        except Exception as e:
            print(f"Error computing alignment for {fname}: {e}")
    if not scores:
        raise ValueError("No valid pairs found for alignment computation")
    print(f"Computed {len(scores)} alignment scores")
    return np.array(scores), filenames


def extract_labels(features_dict):
    labels = []
    for fname, data in features_dict.items():
        label = data.get("label", None)
        if label in [0, 1]:
            labels.append(label)
        else:
            print(f"Skipped invalid label in {fname}")
            labels.append(None)
    return np.array(labels)


def threshold_evaluation(train_scores, train_labels, test_scores, test_labels, config=None, save_output=True):
    valid_train = [i for i, l in enumerate(train_labels) if l is not None]
    train_scores_clean = train_scores[valid_train]
    train_labels_clean = train_labels[valid_train]

    auc_original = roc_auc_score(train_labels_clean, train_scores_clean)
    auc_flipped = roc_auc_score(train_labels_clean, -train_scores_clean)
    
    # Check flip
    if auc_flipped > auc_original:
        print(f"Flipping scores (AUC improved from {auc_original:.3f} to {auc_flipped:.3f})")
        train_scores_clean = -train_scores_clean
        test_scores = -test_scores
        flipped = True
        auc_used = auc_flipped
    else:
        print(f"Using original scores (AUC = {auc_original:.3f})")
        flipped = False
        auc_used = auc_original

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("thresholding_output", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # ROC + threshold
    roc_path = os.path.join(output_dir, "roc_curve.png")
    threshold = plot_roc_curve(train_labels_clean, train_scores_clean, out_path=roc_path)
    print(f"Selected threshold: {threshold:.3f}")

    metrics = evaluate_predictions(test_labels, test_scores, threshold=threshold)

    def to_serializable(d):
        return {k: (v.item() if isinstance(v, (np.generic, np.ndarray)) else v) for k, v in d.items()}

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(to_serializable(metrics), f, indent=4)

    if config is not None:
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    log_summary = (
        f"Thresholding Summary - {timestamp}\n"
        f"AUC Used: {auc_used:.3f}\n"
        f"Flipped: {flipped}\n"
        f"Selected Threshold: {threshold:.3f}\n"
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall: {metrics['recall']:.4f}\n"
        f"F1 Score: {metrics['f1_score']:.4f}\n"
    )
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(log_summary)

    print(f"\nAll thresholding artifacts saved to: {output_dir}")
    return metrics
