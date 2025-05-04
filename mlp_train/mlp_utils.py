import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# 1. Flexible Dataset Class
class AVDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        if isinstance(features, dict):
            self.visual_feats = torch.FloatTensor(features['visual'])
            self.audio_feats = torch.FloatTensor(features['audio'])
            self.feats = torch.cat([self.visual_feats, self.audio_feats], dim=1)
        else:
            self.feats = torch.FloatTensor(features)

        self.labels = torch.FloatTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {
            'features': self.feats[idx],
            'label': self.labels[idx]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

# 2. MLP Model Definition
class SyncDetectorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))  # No Sigmoid here!

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

# 3. Training Function
def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25):
    metrics = {'train': {'loss': [], 'acc': [], 'f1': []},
               'val': {'loss': [], 'acc': [], 'f1': []}}

    best_f1 = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            all_preds = []
            all_labels = []

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
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = accuracy_score(all_labels, all_preds)
            epoch_f1 = f1_score(all_labels, all_preds, average='binary')

            metrics[phase]['loss'].append(epoch_loss)
            metrics[phase]['acc'].append(epoch_acc)
            metrics[phase]['f1'].append(epoch_f1)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')

            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                torch.save(model.state_dict(), 'best_model.pth')

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

    log_content = f"""
    Training Summary - {timestamp}
    --------------------------------
    Best Validation F1: {max(metrics['val']['f1']):.4f}
    Final Training F1: {metrics['train']['f1'][-1]:.4f}
    Training Duration: {config['epochs']} epochs

    Configuration:
    {json.dumps(config, indent=4)}

    Model Architecture:
    {str(model)}
    """
    with open(os.path.join(output_dir, 'training_log.txt'), 'w') as f:
        f.write(log_content)
    print(f"\nAll training artifacts saved to: {output_dir}")

def plot_training_metrics(metrics, output_path):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train']['loss'], label='Train')
    plt.plot(metrics['val']['loss'], label='Validation')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(metrics['train']['f1'], label='Train')
    plt.plot(metrics['val']['f1'], label='Validation')
    plt.title('Training & Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
