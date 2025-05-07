import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.thresholding_utils import compute_alignment_score, compute_all_alignment_scores, extract_labels, threshold_evaluation
from utils.dataloader_utils import unified_split, load_combined_npz_features

# === Configuration ===
npz_dir = r"D:\lenovo\mia_final_project\preprocessed_output"
method = 'cross_corr'  # Alignment method: cross_corr, dynamic_time_warp, or pearson
print(f"🚀 Running threshold-based alignment model using {method} method")

# === Unified Train/Test Split ===
train_files, test_files = unified_split(npz_dir, test_size=0.3, seed=42)
print(f"📊 Split data into {len(train_files)} training and {len(test_files)} testing files")

# === Load and Organize Features ===
all_features = load_combined_npz_features(npz_dir)
print(f"✅ Loaded features for {len(all_features)} files")

train_features = {
    os.path.basename(p): all_features[os.path.basename(p)]
    for p in train_files if os.path.basename(p) in all_features
}
test_features = {
    os.path.basename(p): all_features[os.path.basename(p)]
    for p in test_files if os.path.basename(p) in all_features
}

print(f"✅ Organized {len(train_features)} training and {len(test_features)} testing features")

# === Compute Alignment Scores ===
train_scores, train_filenames = compute_all_alignment_scores(train_features, method=method)
test_scores, test_filenames = compute_all_alignment_scores(test_features, method=method)

# === Extract Labels ===
train_labels = extract_labels(train_features)
test_labels = extract_labels(test_features)

print(f"✅ Extracted labels for {np.sum(train_labels == 0)} negative and {np.sum(train_labels == 1)} positive training samples")
print(f"✅ Extracted labels for {np.sum(test_labels == 0)} negative and {np.sum(test_labels == 1)} positive testing samples")

# === Configuration for saving results ===
config = {
    "alignment_method": method,
    "train_size": len(train_features),
    "test_size": len(test_features),
    "npz_directory": npz_dir
}

# === Run Evaluation ===
print("\n🔍 Running threshold evaluation...")
metrics = threshold_evaluation(
    train_scores, train_labels, 
    test_scores, test_labels,
    config=config, save_output=True
)

# === Print Final Metrics ===
print("\n=== Evaluation Metrics ===")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")