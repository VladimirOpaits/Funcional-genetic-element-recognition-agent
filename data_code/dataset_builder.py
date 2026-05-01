import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Tuple


def create_sliding_windows(embeddings_dict: Dict[str, torch.Tensor], label: int, window_size: int = 32, stride: int = 16):
    windows = []
    labels = []

    for _, embedding in embeddings_dict.items():
        if embedding.dim() == 3:
            embedding = embedding.squeeze(0)

        seq_len = embedding.shape[0]

        if seq_len < window_size:
            windows.append(embedding.mean(dim=0))
            labels.append(label)
        else:
            for start in range(0, seq_len - window_size + 1, stride):
                window = embedding[start:start + window_size]
                windows.append(window.mean(dim=0))
                labels.append(label)

    return windows, labels


def build_dataset(
    class_embeddings: Dict[str, Dict[str, torch.Tensor]],
    window_size: int = 32,
    stride: int = 16,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, torch.Tensor, Dict[str, int]]:
    label_map = {name: idx for idx, name in enumerate(class_embeddings.keys())}
    generator = torch.Generator().manual_seed(seed)

    train_X, train_y = [], []
    val_X, val_y = [], []

    for name, emb_dict in class_embeddings.items():
        label = label_map[name]
        windows, labels = create_sliding_windows(emb_dict, label, window_size, stride)

        if not windows:
            continue

        X = torch.stack(windows)
        y = torch.tensor(labels, dtype=torch.long)

        n = len(X)
        n_val = max(1, int(val_split * n)) if n > 1 else 0
        n_train = n - n_val

        perm = torch.randperm(n, generator=generator)
        train_idx, val_idx = perm[:n_train], perm[n_train:]

        train_X.append(X[train_idx])
        train_y.append(y[train_idx])
        if n_val > 0:
            val_X.append(X[val_idx])
            val_y.append(y[val_idx])

    train_X = torch.cat(train_X)
    train_y = torch.cat(train_y)
    val_X = torch.cat(val_X) if val_X else torch.empty(0, train_X.shape[1])
    val_y = torch.cat(val_y) if val_y else torch.empty(0, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=batch_size, shuffle=False)

    num_classes = len(label_map)
    counts = torch.bincount(train_y, minlength=num_classes).float()
    total = counts.sum()
    class_weights = total / (counts + 1e-8)
    class_weights = class_weights / class_weights.sum() * num_classes

    print(f"Label map: {label_map}")
    print(f"Train: {len(train_X)}, Val: {len(val_X)}")
    print(f"Train distribution: {dict(zip(label_map.keys(), counts.int().tolist()))}")
    print(f"Class weights: {dict(zip(label_map.keys(), [round(w, 3) for w in class_weights.tolist()]))}")

    return train_loader, val_loader, class_weights, label_map
