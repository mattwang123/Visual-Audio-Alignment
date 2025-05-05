import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import os

# Set matplotlib to non-interactive mode
plt.ioff()

def load_data_from_directories(audio_dir, visual_dir):
    """
    Load audio and visual features from directories containing .npy files
    Returns dictionaries mapping filenames to feature arrays
    """
    print(f"Loading audio features from directory: {audio_dir}")
    print(f"Loading visual features from directory: {visual_dir}")
    
    # Check if directories exist
    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"Audio features directory not found: {audio_dir}")
    if not os.path.isdir(visual_dir):
        raise FileNotFoundError(f"Visual features directory not found: {visual_dir}")
    
    # Load all .npy files from directories
    audio_features = {}
    visual_features = {}
    
    # Process audio features
    for filename in os.listdir(audio_dir):
        if filename.endswith('.npy'):
            file_path = os.path.join(audio_dir, filename)
            try:
                feature = np.load(file_path, allow_pickle=True)
                audio_features[filename] = feature
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    # Process visual features
    for filename in os.listdir(visual_dir):
        if filename.endswith('.npy'):
            file_path = os.path.join(visual_dir, filename)
            try:
                feature = np.load(file_path, allow_pickle=True)
                visual_features[filename] = feature
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(audio_features)} audio feature files")
    print(f"Loaded {len(visual_features)} visual feature files")
    
    return audio_features, visual_features

def extract_labels_from_filenames(filenames):
    """
    Extract labels (good/bad) from filenames
    Returns binary labels: 0 for "good", 1 for "bad"
    """
    labels = []
    for filename in filenames:
        # Extract the label from the filename
        if isinstance(filename, str):
            if "bad" in filename.lower():
                labels.append(1)  # Anomaly
            elif "good" in filename.lower():
                labels.append(0)  # Normal
            else:
                # If no clear label in filename, append None or placeholder
                print(f"Warning: No clear label found in filename: {filename}")
                labels.append(None)
    
    # Count valid labels
    valid_labels = [l for l in labels if l is not None]
    if valid_labels:
        anomalies = sum(1 for l in valid_labels if l == 1)
        normal = sum(1 for l in valid_labels if l == 0)
        print(f"Extracted {len(valid_labels)} labels: {anomalies} anomalies ({anomalies/len(valid_labels)*100:.1f}%), {normal} normal ({normal/len(valid_labels)*100:.1f}%)")
    else:
        print("Warning: No valid labels extracted from filenames")
    
    return np.array(labels)

def compute_alignment_score(audio_feat, visual_feat):
    """
    Compute alignment score between audio and visual features
    Higher score indicates worse alignment (more likely to be an anomaly)
    """
    # Handle different shapes of features
    if isinstance(audio_feat, np.ndarray) and isinstance(visual_feat, np.ndarray):
        # Flatten arrays if they're multi-dimensional
        if audio_feat.ndim > 1:
            audio_feat = audio_feat.flatten()
        if visual_feat.ndim > 1:
            visual_feat = visual_feat.flatten()
        
        # Handle different length features
        min_len = min(len(audio_feat), len(visual_feat))
        
        # Calculate normalized Euclidean distance
        # We normalize to account for different feature scales
        if min_len > 0:
            # Truncate to equal lengths
            audio_trunc = audio_feat[:min_len]
            visual_trunc = visual_feat[:min_len]
            
            # Normalize to 0-1 range if values exist and range is non-zero
            if np.ptp(audio_trunc) > 0:
                audio_trunc = (audio_trunc - np.min(audio_trunc)) / np.ptp(audio_trunc)
            if np.ptp(visual_trunc) > 0:
                visual_trunc = (visual_trunc - np.min(visual_trunc)) / np.ptp(visual_trunc)
            
            # Compute Euclidean distance between normalized features
            alignment_score = np.linalg.norm(audio_trunc - visual_trunc) / np.sqrt(min_len)
            
        else:
            alignment_score = 1.0  # Maximum misalignment if no features to compare
    else:
        # If inputs aren't arrays, default to maximum misalignment
        alignment_score = 1.0
    
    return alignment_score

def compute_all_alignment_scores(audio_features, visual_features):
    """
    Compute alignment scores for all pairs of audio and visual features
    For directory-based data: audio_features and visual_features are dictionaries with filenames as keys
    """
    scores = []
    filenames = []
    
    # Match files from both directories
    if isinstance(audio_features, dict) and isinstance(visual_features, dict):
        # Get common filenames (files that exist in both audio and visual directories)
        # We'll first normalize filenames to handle potential differences in naming patterns
        
        # Extract base filenames without extensions
        audio_bases = {os.path.splitext(fname)[0]: fname for fname in audio_features.keys()}
        visual_bases = {os.path.splitext(fname)[0]: fname for fname in visual_features.keys()}
        
        # Find common base names
        common_bases = set(audio_bases.keys()).intersection(set(visual_bases.keys()))
        print(f"Found {len(common_bases)} matching files in both directories")
        
        # For each matching file pair, compute alignment score
        for base in common_bases:
            audio_file = audio_bases[base]
            visual_file = visual_bases[base]
            
            try:
                score = compute_alignment_score(audio_features[audio_file], visual_features[visual_file])
                scores.append(score)
                # Keep the full filename with extension for label extraction
                filenames.append(audio_file)
            except Exception as e:
                print(f"Error computing alignment for {base}: {e}")
    else:
        raise ValueError("Expected dictionaries mapping filenames to features")
    
    if not scores:
        raise ValueError("No valid audio-visual pairs found for alignment computation")
    
    return np.array(scores), filenames

def find_optimal_threshold(scores, labels, plot=False):  # Changed default to False
    """
    Find optimal threshold for anomaly detection
    """
    # Filter out None values
    valid_indices = [i for i, label in enumerate(labels) if label is not None]
    valid_scores = scores[valid_indices]
    valid_labels = labels[valid_indices]
    
    if len(valid_scores) == 0 or len(valid_labels) == 0:
        raise ValueError("No valid scores or labels found")
    
    # Use ROC curve to find optimal threshold
    fpr, tpr, thresholds = roc_curve(valid_labels, valid_scores)
    
    # Find threshold that maximizes Youden's J statistic (TPR - FPR)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]
    
    if plot:
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc(fpr, tpr):.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.scatter(fpr[best_idx], tpr[best_idx], marker='o', color='red', 
                   label=f'Optimal Threshold ({optimal_threshold:.3f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Anomaly Detection')
        plt.legend()
        plt.grid(True)
        # Save the plot instead of showing it
        plt.savefig('roc_curve.png')
        plt.close()  # Close the figure to free memory
        print("ROC curve saved to 'roc_curve.png'")
    
    return optimal_threshold

def evaluate_model(scores, labels, threshold):
    """
    Evaluate model performance using the given threshold
    """
    # Predict using threshold
    predicted_labels = (scores >= threshold).astype(int)
    
    # Filter out None values
    valid_indices = [i for i, label in enumerate(labels) if label is not None]
    valid_preds = predicted_labels[valid_indices]
    valid_labels = labels[valid_indices]
    
    if len(valid_preds) == 0 or len(valid_labels) == 0:
        return {"error": "No valid predictions or labels found"}
    
    # Calculate metrics
    accuracy = accuracy_score(valid_labels, valid_preds)
    precision = precision_score(valid_labels, valid_preds, zero_division=0)
    recall = recall_score(valid_labels, valid_preds, zero_division=0)
    f1 = f1_score(valid_labels, valid_preds, zero_division=0)
    
    # Create confusion matrix
    tn = sum((valid_labels == 0) & (valid_preds == 0))
    fp = sum((valid_labels == 0) & (valid_preds == 1))
    fn = sum((valid_labels == 1) & (valid_preds == 0))
    tp = sum((valid_labels == 1) & (valid_preds == 1))
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "threshold": threshold
    }
    
    return metrics

def display_metrics(metrics):
    """
    Display evaluation metrics
    """
    print("\n=== Evaluation Metrics ===")
    print(f"Threshold: {metrics['threshold']:.3f}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    
    print("\n=== Confusion Matrix ===")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"True Negatives: {metrics['true_negatives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    
    # Visual confusion matrix
    cm = np.array([
        [metrics['true_negatives'], metrics['false_positives']], 
        [metrics['false_negatives'], metrics['true_positives']]
    ])
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Normal', 'Anomaly']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add labels
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    # Save the plot instead of showing it
    plt.savefig('confusion_matrix.png')
    plt.close()  # Close the figure to free memory
    print("Confusion matrix saved to 'confusion_matrix.png'")

def main():
    try:
        # Print progress message
        print("Starting audio-visual anomaly detection...")
        
        # Directories containing feature files
        visual_features_dir = r"D:\lenovo\mia_final_project\Visual-Audio-Alignment\visual_features"
        audio_features_dir = r"D:\lenovo\mia_final_project\Visual-Audio-Alignment\audio_features" 
        
        # Load features from directories
        print("Loading feature files...")
        audio_features, visual_features = load_data_from_directories(audio_features_dir, visual_features_dir)
        
        # Compute alignment scores
        print("Computing alignment scores...")
        alignment_scores, filenames = compute_all_alignment_scores(audio_features, visual_features)
        print(f"Computed {len(alignment_scores)} alignment scores")
        
        # Extract labels from filenames
        print("Extracting labels from filenames...")
        labels = extract_labels_from_filenames(filenames)
        
        # Split data into training (70%) and test (30%) sets
        print("Splitting data into training and test sets...")
        indices = np.arange(len(alignment_scores))
        train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=42)
        
        train_scores = alignment_scores[train_indices]
        train_labels = labels[train_indices]
        test_scores = alignment_scores[test_indices]
        test_labels = labels[test_indices]
        
        print(f"Split data into {len(train_scores)} training and {len(test_scores)} test samples")
        
        # Find optimal threshold using training data
        print("Finding optimal threshold...")
        optimal_threshold = find_optimal_threshold(train_scores, train_labels, plot=True)  # Set to True to save ROC curve
        print(f"Optimal threshold: {optimal_threshold:.3f}")
        
        # Evaluate on test set
        print("Evaluating model on test set...")
        test_metrics = evaluate_model(test_scores, test_labels, optimal_threshold)
        display_metrics(test_metrics)
        
        # Return success
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()