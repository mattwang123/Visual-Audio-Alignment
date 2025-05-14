import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import os
import json
import matplotlib.pyplot as plt

# Hyperparameters
HIDDEN_DIMS = [512, 128, 64]
DROPOUT = 0.4
LR = 8e-7
WEIGHT_DECAY = 0
NUM_EPOCHS = 100
BATCH_SIZE = 128

import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.activation = Swish()

    def forward(self, x):
        return self.activation(x + self.block(x))

class SyncDetectorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT):
        super().__init__()
        layers = []
        prev_dim = input_dim

        layers.append(nn.Linear(prev_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(Swish())
        layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dims[0]

        for h_dim in hidden_dims[1:]:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(ResidualBlock(h_dim, dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))  # For BCEWithLogitsLoss

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=NUM_EPOCHS):
    metrics = {
        'train': {'loss': [], 'acc': [], 'f1': [], 'precision': [], 'recall': []},
        'val': {'loss': [], 'acc': [], 'f1': [], 'precision': [], 'recall': []}
    }

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
            epoch_precision = precision_score(all_labels, all_preds)
            epoch_recall = recall_score(all_labels, all_preds)

            metrics[phase]['loss'].append(epoch_loss)
            metrics[phase]['acc'].append(epoch_acc)
            metrics[phase]['f1'].append(epoch_f1)
            metrics[phase]['precision'].append(epoch_precision)
            metrics[phase]['recall'].append(epoch_recall)
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f} Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f}")

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
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    axes = axes.flatten()

    metric_names = ['loss', 'acc', 'f1', 'precision', 'recall']
    titles = ['Loss', 'Accuracy', 'F1 Score', 'Precision', 'Recall']
    ylabels = ['Loss', 'Accuracy', 'F1 Score', 'Precision', 'Recall']

    for i, metric in enumerate(metric_names):
        axes[i].plot(metrics['train'][metric], label='Train', linewidth=2)
        axes[i].plot(metrics['val'][metric], label='Val', linewidth=2)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(ylabels[i])
        axes[i].legend()
        axes[i].grid(True)

    if len(metric_names) < len(axes):
        axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=500)
    plt.close()
