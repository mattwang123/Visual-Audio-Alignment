import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import os
import json
import matplotlib.pyplot as plt

from utils.evaluation_utils import evaluate_predictions, plot_confusion_matrix, plot_roc_curve

# === Hyperparameter Macros ===
HIDDEN_DIMS = [1024, 512, 128]
DROPOUT = 0.4
LR = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50
BATCH_SIZE = 64
SCHEDULER_STEP = 10
SCHEDULER_GAMMA = 0.5

class SyncDetectorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=NUM_EPOCHS):
    metrics = {'train': {'loss': [], 'acc': [], 'f1': []},
               'val': {'loss': [], 'acc': [], 'f1': []}}
    best_f1 = 0.0
    best_preds = []
    best_labels = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}\n{"-" * 10}')
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            all_preds, all_labels, all_probs = [], [], []

            for batch in dataloaders[phase]:
                features = batch['features'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    logits = model(features)
                    loss = criterion(logits, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                running_loss += loss.item() * features.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = accuracy_score(all_labels, all_preds)
            epoch_f1 = f1_score(all_labels, all_preds)

            metrics[phase]['loss'].append(epoch_loss)
            metrics[phase]['acc'].append(epoch_acc)
            metrics[phase]['f1'].append(epoch_f1)
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}")

            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                torch.save(model.state_dict(), 'best_model.pth')
                best_preds = all_probs.copy()
                best_labels = all_labels.copy()

        scheduler.step()

    print("\nGenerating evaluation plots from best validation performance...")
    threshold = plot_roc_curve(best_labels, best_preds, out_path="mlp_roc_curve.png")
    best_metrics = evaluate_predictions(best_labels, best_preds, threshold=threshold)
    plot_confusion_matrix(best_metrics, out_path="mlp_confusion_matrix.png")

    return metrics, model

def save_model_and_metrics(model, optimizer, metrics, config, output_dir, timestamp):
    os.makedirs(output_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': config['epochs'],
        'metrics': metrics,
        'config': config
    }, os.path.join(output_dir, 'full_training_state.pth'))

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    with open(os.path.join(output_dir, 'model_architecture.txt'), 'w') as f:
        f.write(str(model))

    summary = (
        f"Training Summary - {timestamp}\n"
        f"Best Validation F1: {max(metrics['val']['f1']):.4f}\n"
        f"Final Training F1: {metrics['train']['f1'][-1]:.4f}\n"
        f"Training Duration: {config['epochs']} epochs\n"
    )
    with open(os.path.join(output_dir, 'training_log.txt'), 'w') as f:
        f.write(summary)
    print(f"\nAll training artifacts saved to: {output_dir}")

def plot_training_metrics(metrics, output_path):
    plt.figure(figsize=(18, 4))

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(metrics['train']['loss'], label='Train')
    plt.plot(metrics['val']['loss'], label='Val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(metrics['train']['acc'], label='Train')
    plt.plot(metrics['val']['acc'], label='Val')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot F1 Score
    plt.subplot(1, 3, 3)
    plt.plot(metrics['train']['f1'], label='Train')
    plt.plot(metrics['val']['f1'], label='Val')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()