import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.nt_model import NucleotideTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "data_code" / "processed"
SHARD_DIR = PROJECT_ROOT / "data_code" / "annotated_shards"

CLASSES = ["INT", "RT", "RN", "NEG"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}

DOMAIN_FASTA_MAP = {"INT": "INT", "RT": "RT", "RNaseH": "RN"}
SHARD_CLASSES = ["INT", "RT", "RN", "GAG"]
SHARD_TO_LABEL = {i: LABEL_MAP[c] for i, c in enumerate(SHARD_CLASSES) if c in LABEL_MAP}

WINDOW = 64
STRIDE = 16
SEED = 42
HOLDOUT = {"3280885232", "3280926678"}
NEG_PER_ACC = 500
VAL_SPLIT = 0.2


def embed_fastas(
    paths: List[Path],
    out_path: Path,
    nt: NucleotideTransformer,
) -> Dict[str, torch.Tensor]:
    store: Dict[str, torch.Tensor] = {}
    for fp in paths:
        added = 0
        for record in SeqIO.parse(str(fp), 'fasta'):
            seq = str(record.seq)
            if not seq:
                continue
            key = f"{fp.stem}:{record.id}"
            if key in store:
                key = f"{key}#{added}"
            store[key] = nt.embed(seq)
            added += 1
        print(f"    {fp.name}: +{added}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(store, out_path)
    return store


def ensure_canonical_embeddings() -> Dict[str, Dict[str, torch.Tensor]]:
    nt: Optional[NucleotideTransformer] = None
    out: Dict[str, Dict[str, torch.Tensor]] = {}

    for stem, cls in DOMAIN_FASTA_MAP.items():
        fasta = DATA_DIR / "retro_domains" / f"{stem}.fasta"
        if not fasta.exists():
            print(f"  SKIP {cls}: нет {fasta}")
            continue
        out_path = PROCESSED_DIR / f"{cls}_embeddings.pt"
        if out_path.exists():
            print(f"  [cache] {out_path.name}")
            out[cls] = torch.load(out_path, map_location='cpu', weights_only=False)
            continue
        if nt is None:
            print("  Загрузка NucleotideTransformer...")
            nt = NucleotideTransformer()
        print(f"  Embedding {cls} <- {fasta.name}")
        out[cls] = embed_fastas([fasta], out_path, nt)

    neg_dir = DATA_DIR / "negatives"
    neg_paths = sorted(list(neg_dir.glob("*.fasta")) + list(neg_dir.glob("*.fa")))
    if neg_paths:
        out_path = PROCESSED_DIR / "NEG_embeddings.pt"
        if out_path.exists():
            print(f"  [cache] {out_path.name}")
            out["NEG"] = torch.load(out_path, map_location='cpu', weights_only=False)
        else:
            if nt is None:
                print("  Загрузка NucleotideTransformer...")
                nt = NucleotideTransformer()
            print(f"  Embedding NEG <- {[p.name for p in neg_paths]}")
            out["NEG"] = embed_fastas(neg_paths, out_path, nt)
    else:
        print("  SKIP NEG: нет fasta в data/negatives/")

    return out


def _usable_counts(counts: np.ndarray) -> np.ndarray:
    return np.array([counts[i] if i in SHARD_TO_LABEL else 0 for i in range(len(counts))])


def collect_shard_data(
    accession: str,
    window: int,
    stride: int,
    neg_per_acc: Optional[int] = None,
    include_background_as_neg: bool = False,
) -> Tuple[List[torch.Tensor], List[int], List[float]]:
    """include_background_as_neg=True: фоновые окна получают метку NEG (для holdout).
    Иначе берём фон отдельно и сабсемплируем neg_per_acc штук."""
    Xs, ys, positions = [], [], []
    bg_X, bg_pos = [], []

    for p in sorted(SHARD_DIR.glob(f"{accession}_*.pt")):
        shard = torch.load(p, map_location='cpu', weights_only=False)
        for chunk in shard['chunks']:
            offsets = chunk['token_offsets']
            keep = torch.tensor([s != e for s, e in offsets], dtype=torch.bool)
            emb = chunk['embeddings'][keep].float()
            lab = chunk['labels'][keep]
            centers = np.array([(s + e) / 2 for s, e in offsets if s != e])
            T = emb.shape[0]
            if T < window:
                continue

            for start in range(0, T - window + 1, stride):
                counts = lab[start:start + window].sum(dim=0).numpy()
                usable = _usable_counts(counts)
                center = centers[start:start + window].mean()
                pooled = emb[start:start + window].mean(dim=0)

                if usable.sum() > 0:
                    Xs.append(pooled)
                    ys.append(SHARD_TO_LABEL[int(usable.argmax())])
                    positions.append(center)
                else:
                    if include_background_as_neg:
                        Xs.append(pooled)
                        ys.append(LABEL_MAP['NEG'])
                        positions.append(center)
                    else:
                        bg_X.append(pooled)
                        bg_pos.append(center)
        del shard

    if not include_background_as_neg and neg_per_acc and bg_X:
        rng = np.random.default_rng(SEED + abs(hash(accession)) % 10_000)
        n = min(neg_per_acc, len(bg_X))
        idx = rng.choice(len(bg_X), n, replace=False)
        for i in idx:
            Xs.append(bg_X[i])
            ys.append(LABEL_MAP['NEG'])
            positions.append(bg_pos[i])

    return Xs, ys, positions


def windows_from_canonical(
    emb_dict: Dict[str, torch.Tensor],
    label_idx: int,
    window: int,
    stride: int,
) -> Tuple[List[torch.Tensor], List[int]]:
    Xs, ys = [], []
    for emb in emb_dict.values():
        if emb.dim() == 3:
            emb = emb.squeeze(0)
        emb = emb.float()
        T = emb.shape[0]
        if T < window:
            Xs.append(emb.mean(dim=0))
            ys.append(label_idx)
            continue
        for start in range(0, T - window + 1, stride):
            Xs.append(emb[start:start + window].mean(dim=0))
            ys.append(label_idx)
    return Xs, ys


def build():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Шаг 1: эмбеддинги retro_domains + negatives ===")
    canonical = ensure_canonical_embeddings()

    print("\n=== Шаг 2: проверка шардов ===")
    shard_files = sorted(SHARD_DIR.glob('*.pt'))
    if not shard_files:
        raise SystemExit(f"Нет шардов в {SHARD_DIR}. Сначала запусти annotated_processor.py")
    all_acc = sorted({p.name.split('_')[0] for p in shard_files})
    train_acc = [a for a in all_acc if a not in HOLDOUT]
    holdout_acc = [a for a in all_acc if a in HOLDOUT]
    print(f"  Все аксессионы: {all_acc}")
    print(f"  Train+val: {train_acc}")
    print(f"  Holdout:   {holdout_acc}")

    print("\n=== Шаг 3: окна из train-шардов (с фоном как NEG, сабсемпл) ===")
    X_all, y_all = [], []
    for acc in train_acc:
        Xs, ys, _ = collect_shard_data(acc, WINDOW, STRIDE, neg_per_acc=NEG_PER_ACC)
        X_all.extend(Xs); y_all.extend(ys)
        print(f"  {acc}: +{len(Xs)} окон")

    print("\n=== Шаг 4: окна из retro_domains + negatives ===")
    for cls, emb_dict in canonical.items():
        xs, ys = windows_from_canonical(emb_dict, LABEL_MAP[cls], WINDOW, STRIDE)
        X_all.extend(xs); y_all.extend(ys)
        print(f"  {cls}: +{len(xs)} окон ({len(emb_dict)} последовательностей)")

    X_all = torch.stack(X_all)
    y_all = torch.tensor(y_all, dtype=torch.long)
    print(f"\nВсего train+val окон: {len(X_all)}")
    for i, cls in enumerate(CLASSES):
        n = (y_all == i).sum().item()
        print(f"  {cls}: {n}  ({100 * n / len(y_all):.1f}%)")

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(X_all), generator=g)
    n_val = int(VAL_SPLIT * len(X_all))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    print("\n=== Шаг 5: holdout (все окна, фон → NEG) ===")
    holdout_data = {}
    for acc in holdout_acc:
        Xs, ys, positions = collect_shard_data(acc, WINDOW, STRIDE, include_background_as_neg=True)
        if not Xs:
            continue
        holdout_data[acc] = {
            "X": torch.stack(Xs),
            "y": torch.tensor(ys, dtype=torch.long),
            "positions": torch.tensor(positions, dtype=torch.float32),
        }
        dist = {cls: int((holdout_data[acc]["y"] == i).sum()) for i, cls in enumerate(CLASSES)}
        print(f"  {acc}: {len(Xs)} окон  {dist}")

    out = {
        "classes": CLASSES,
        "label_map": LABEL_MAP,
        "train_X": X_all[train_idx].contiguous(),
        "train_y": y_all[train_idx].contiguous(),
        "val_X": X_all[val_idx].contiguous(),
        "val_y": y_all[val_idx].contiguous(),
        "holdout": holdout_data,
        "config": {
            "window": WINDOW,
            "stride": STRIDE,
            "seed": SEED,
            "val_split": VAL_SPLIT,
            "holdout_accessions": sorted(HOLDOUT),
            "neg_per_acc": NEG_PER_ACC,
        },
    }

    out_path = PROCESSED_DIR / "dataset.pt"
    torch.save(out, out_path)
    print(f"\nSaved dataset -> {out_path}")
    return out


if __name__ == "__main__":
    build()
