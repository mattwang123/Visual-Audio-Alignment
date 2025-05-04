import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, random_split
from mlp_utils import SyncDetectorMLP, train_model, plot_training_metrics, save_model_and_metrics

# === Path to preprocessed .npz files ===
preprocessed_dir = r"D:\lenovo\mia_final_project\Visual-Audio-Alignment\preprocessed_output"
all_npz_files = [os.path.join(preprocessed_dir, f) for f in os.listdir(preprocessed_dir) if f.endswith(".npz")]

# === Split train/val (80/20) ===
train_ratio = 0.8
train_len = int(train_ratio * len(all_npz_files))
val_len = len(all_npz_files) - train_len
train_files, val_files = random_split(all_npz_files, [train_len, val_len], generator=torch.Generator().manual_seed(42))

# === Custom Dataset for .npz files ===
class PreprocessedAVDataset(Dataset):
    def __init__(self, file_list):
        self.data = []
        self.labels = []

        for path in file_list:
            try:
                arr = np.load(path)
                visual = arr['visual']
                audio = arr['audio']
                label = float(arr['label'])

                # Skip malformed or 1D data
                if visual.ndim != 2 or audio.ndim != 2 or len(visual) == 0 or len(audio) == 0:
                    raise ValueError(f"Invalid shape: visual {visual.shape}, audio {audio.shape}")

                min_len = min(len(visual), len(audio))
                visual, audio = visual[:min_len], audio[:min_len]
                combined = np.concatenate([visual, audio], axis=1)
                pooled = combined.mean(axis=0)

                self.data.append(pooled)
                self.labels.append(label)
            except Exception as e:
                print(f"Skipping {path} due to error: {e}")

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'features': self.data[idx],
            'label': self.labels[idx]
        }

# === Load datasets ===
train_dataset = PreprocessedAVDataset(train_files)
val_dataset = PreprocessedAVDataset(val_files)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=64, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=64)
}

# === Model Config ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = train_dataset[0]['features'].shape[0]
model = SyncDetectorMLP(input_dim=input_dim, hidden_dims=[256, 128], dropout=0.3).to(device)

criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# === Train ===
metrics, model = train_model(model, dataloaders, criterion, optimizer, device, num_epochs=50)

# === Save Results ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"training_output_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

plot_training_metrics(metrics, os.path.join(output_dir, 'training_metrics.png'))
save_model_and_metrics(model, optimizer, metrics, {
    'batch_size': 64,
    'hidden_dims': [256, 128],
    'dropout': 0.3,
    'lr': 1e-3,
    'epochs': 50,
    'weight_decay': 1e-5
}, output_dir, timestamp)
