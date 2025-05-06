import os
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.thresholding_utils import compute_alignment_score, threshold_evaluation
from utils.dataloader_utils import unified_split, load_combined_npz_features

# === Configuration ===
npz_dir = r"D:\lenovo\mia_final_project\preprocessed_output"
print("🚀 Running threshold-based alignment model")

# === Unified Train/Test Split ===
train_files, test_files = unified_split(npz_dir, test_size=0.3, seed=42)

# === Load and Organize Features ===
all_features = load_combined_npz_features(npz_dir)
train_features = {
    os.path.basename(p): all_features[os.path.basename(p)]
    for p in train_files if os.path.basename(p) in all_features
}
test_features = {
    os.path.basename(p): all_features[os.path.basename(p)]
    for p in test_files if os.path.basename(p) in all_features
}

# === Compute Alignment Scores and Labels ===
train_scores = np.array([
    compute_alignment_score(f['audio'], f['visual']) for f in train_features.values()
])
train_labels = np.array([f['label'] for f in train_features.values()])

test_scores = np.array([
    compute_alignment_score(f['audio'], f['visual']) for f in test_features.values()
])
test_labels = np.array([f['label'] for f in test_features.values()])

# === Run Evaluation ===
metrics = threshold_evaluation(train_scores, train_labels, test_scores, test_labels)

# === Print Final Metrics ===
print("\n=== Evaluation Metrics ===")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")