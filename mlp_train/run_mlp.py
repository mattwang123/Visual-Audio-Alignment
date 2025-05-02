import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from datetime import datetime

from mlp_utils import AVDataset, SyncDetectorMLP, train_model, plot_training_metrics, save_model_and_metrics

if __name__ == "__main__":
    # Configurations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = {
        'batch_size': 64,
        'hidden_dims': [256, 128],
        'dropout': 0.3,
        'lr': 1e-3,
        'epochs': 50,
        'weight_decay': 1e-5
    }

    # Simulated data (replace with actual data loading)
    train_features = {
        'visual': np.random.randn(1000, 20),
        'audio': np.random.randn(1000, 15)
    }
    train_labels = np.random.randint(1, 2, 1000)

    val_features = {
        'visual': np.random.randn(200, 20),
        'audio': np.random.randn(200, 15)
    }
    val_labels = np.random.randint(1, 2, 200)

    # Dataset and Dataloader
    train_dataset = AVDataset(train_features, train_labels)
    val_dataset = AVDataset(val_features, val_labels)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True),
        'val': DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    }

    # Model
    input_dim = train_dataset.feats.shape[1]
    model = SyncDetectorMLP(input_dim, hidden_dims=config['hidden_dims'], dropout=config['dropout']).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # Train
    metrics, model = train_model(model, dataloaders, criterion, optimizer, device, num_epochs=config['epochs'])

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"training_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    plot_training_metrics(metrics, os.path.join(output_dir, 'training_metrics.png'))
    save_model_and_metrics(model, optimizer, metrics, config, output_dir, timestamp)
