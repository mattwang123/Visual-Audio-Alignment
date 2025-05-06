import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from utils.mlp_utils import (
    SyncDetectorMLP, train_model, plot_training_metrics, save_model_and_metrics,
    HIDDEN_DIMS, DROPOUT, LR, WEIGHT_DECAY, NUM_EPOCHS, BATCH_SIZE, SCHEDULER_STEP, SCHEDULER_GAMMA
)
from utils.dataloader_utils import unified_split, PreprocessedAVDataset

# === Path to .npz files ===
npz_dir = r"D:\lenovo\mia_final_project\preprocessed_output"
train_files, val_files = unified_split(npz_dir, test_size=0.3, seed=42)

# === Load datasets ===
train_dataset = PreprocessedAVDataset(train_files)
val_dataset = PreprocessedAVDataset(val_files)

# Optional: Label distribution check
def check_label_distribution(dataset, name="Dataset"):
    labels = [sample['label'].item() for sample in dataset]
    counts = np.bincount(np.array(labels, dtype=int))
    total = len(labels)
    print(f"\n📊 {name} Label Distribution:")
    for i, count in enumerate(counts):
        print(f"Label {i}: {count} ({count/total:.2%})")

check_label_distribution(train_dataset, "Training Set")
check_label_distribution(val_dataset, "Validation Set")

# === Dataloaders ===
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=BATCH_SIZE)
}

# === Model Config ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = train_dataset[0]['features'].shape[0]
model = SyncDetectorMLP(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)

# === Train ===
metrics, model = train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=NUM_EPOCHS)

# === Save Results ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"training_output_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

plot_training_metrics(metrics, os.path.join(output_dir, 'training_metrics.png'))

save_model_and_metrics(model, optimizer, metrics, {
    'batch_size': BATCH_SIZE,
    'hidden_dims': HIDDEN_DIMS,
    'dropout': DROPOUT,
    'lr': LR,
    'epochs': NUM_EPOCHS,
    'weight_decay': WEIGHT_DECAY,
    'scheduler_step': SCHEDULER_STEP,
    'scheduler_gamma': SCHEDULER_GAMMA
}, output_dir, timestamp)
