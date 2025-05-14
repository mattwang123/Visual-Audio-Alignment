import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def load_combined_npz_features(npz_dir):
    features = {}
    for fname in os.listdir(npz_dir):
        if fname.endswith(".npz"):
            path = os.path.join(npz_dir, fname)
            try:
                arr = np.load(path)
                if 'audio' in arr and 'visual' in arr and 'label' in arr:
                    features[fname] = {
                        'audio': arr['audio'],
                        'visual': arr['visual'],
                        'label': int(arr['label'])
                    }
            except Exception as e:
                print(f"Failed to load {fname}: {e}")
    print(f"Loaded {len(features)} combined feature files.")
    return features

def unified_split(npz_dir, test_size=0.3, seed=42):
    all_files = sorted([
        os.path.join(npz_dir, f)
        for f in os.listdir(npz_dir)
        if f.endswith('.npz')
    ])
    
    train_files, test_files = train_test_split(
        all_files, test_size=test_size, random_state=seed
    )
    
    return train_files, test_files

class PreprocessedAVDataset(Dataset):
    def __init__(self, file_list, expected_dim=9229):
        self.data = []
        self.labels = []

        for path in file_list:
            try:
                arr = np.load(path)
                visual = arr['visual']
                audio = arr['audio']
                label = float(arr['label'])

                if visual.ndim != 2 or audio.ndim != 2 or len(visual) == 0 or len(audio) == 0:
                    continue

                min_len = min(len(visual), len(audio))
                concat = np.concatenate([visual[:min_len], audio[:min_len]], axis=1)
                pooled = np.mean(concat, axis=0)

                if pooled.shape[0] != expected_dim:
                    #print(f"Skipping {path}: Unexpected pooled feature dimension {pooled.shape[0]}")
                    continue

                if np.isnan(pooled).any() or np.isinf(pooled).any():
                    #print(f"NaN/Inf detected in file {path}")
                    continue

                self.data.append(pooled)
                self.labels.append(label)

            except Exception as e:
                print(f"Skipping {path}: {e}")

        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'features': self.data[idx],
            'label': self.labels[idx]
        }
