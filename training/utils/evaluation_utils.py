import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc
)

plt.ioff()

def evaluate_predictions(y_true, y_pred, threshold=None):
    predicted = (y_pred >= threshold).astype(int) if threshold is not None else y_pred

    accuracy = accuracy_score(y_true, predicted)
    precision = precision_score(y_true, predicted, zero_division=0)
    recall = recall_score(y_true, predicted, zero_division=0)
    f1 = f1_score(y_true, predicted, zero_division=0)

    tn = sum((y_true == 0) & (predicted == 0))
    fp = sum((y_true == 0) & (predicted == 1))
    fn = sum((y_true == 1) & (predicted == 0))
    tp = sum((y_true == 1) & (predicted == 1))

    return {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1,
        "true_positives": tp, "false_positives": fp, "true_negatives": tn,
        "false_negatives": fn, "threshold": threshold
    }

def plot_confusion_matrix(metrics, out_path="confusion_matrix.png"):
    cm = np.array([
        [metrics['true_negatives'], metrics['false_positives']],
        [metrics['false_negatives'], metrics['true_positives']]
    ])

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ['Normal', 'Anomaly'])
    plt.yticks([0, 1], ['Normal', 'Anomaly'])

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center',
                     color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"📊 Confusion matrix saved to '{out_path}'")

def plot_roc_curve(y_true, y_score, out_path="roc_curve.png"):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    best_idx = np.argmax(tpr - fpr)
    best_thresh = thresholds[best_idx]

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.2f})')
    plt.scatter(fpr[best_idx], tpr[best_idx], c='r', label=f'Best: {best_thresh:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()
    print(f"📊 ROC curve saved to '{out_path}'")

    return best_thresh
