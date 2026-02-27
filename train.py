#!/usr/bin/env python3
"""Train a fractal adapter to transform 128d SCimilarity embeddings into
hierarchy-aligned fractal embeddings.

Uses FractalHeadV5 from the fractal-embeddings submodule. The adapter learns to
produce 128d output (2 scales x 64d) where the first 64d prefix captures
coarse system-level structure and the full 128d captures fine subtype detail.

Training uses the same loss recipe as fractal_v5: contrastive + margin +
classification with prefix supervision and block dropout.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import FractalHeadV5 from submodule
sys.path.insert(0, str(Path(__file__).parent / "fractal-embeddings" / "moonshot-fractal-embeddings" / "src"))
from fractal_v5 import FractalHeadV5

INPUT_PATH = Path("data/scimilarity_embeddings.parquet")
OUTPUT_MODEL_PATH = Path("data/fractal_adapter.pt")

# Fractal architecture
NUM_SCALES = 2
SCALE_DIM = 64
INPUT_DIM = 128

# Loss hyperparameters (from V5Trainer, adapted for 2 scales)
PREFIX_PROBS = [0.6, 0.4]  # heavier weight on coarse prefix
BLOCK_KEEP_PROBS = [0.95, 0.7]  # block dropout per scale
PREFIX_WEIGHT = 0.6
MARGIN_WEIGHT = 0.5
CLASS_WEIGHT = 1.0
TEMPERATURE = 0.07
MARGIN = 0.2


class CellEmbeddingDataset(Dataset):
    """Dataset of pre-computed cell embeddings with hierarchical labels.

    Supports contrastive sampling: each __getitem__ returns an anchor plus
    L0-positive (same system) and L1-positive (same subtype) embeddings.
    """

    def __init__(self, embeddings, l0_labels, l1_labels):
        self.embeddings = embeddings  # (N, 128) float32 tensor
        self.l0_labels = l0_labels    # (N,) int64 tensor
        self.l1_labels = l1_labels    # (N,) int64 tensor

        # Build indices for contrastive sampling
        self.l0_index = {}
        self.l1_index = {}
        for i in range(len(self.l0_labels)):
            l0 = self.l0_labels[i].item()
            l1 = self.l1_labels[i].item()
            self.l0_index.setdefault(l0, []).append(i)
            self.l1_index.setdefault(l1, []).append(i)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        anchor = self.embeddings[idx]
        l0 = self.l0_labels[idx].item()
        l1 = self.l1_labels[idx].item()

        # L0 positive (same system)
        l0_indices = self.l0_index[l0]
        l0_pos_idx = random.choice(l0_indices)
        while l0_pos_idx == idx and len(l0_indices) > 1:
            l0_pos_idx = random.choice(l0_indices)

        # L1 positive (same subtype)
        l1_indices = self.l1_index[l1]
        l1_pos_idx = random.choice(l1_indices)
        while l1_pos_idx == idx and len(l1_indices) > 1:
            l1_pos_idx = random.choice(l1_indices)

        return {
            "anchor": anchor,
            "l0_pos": self.embeddings[l0_pos_idx],
            "l1_pos": self.embeddings[l1_pos_idx],
            "l0_label": self.l0_labels[idx],
            "l1_label": self.l1_labels[idx],
        }


def load_data(max_samples, val_ratio, seed):
    """Load embeddings and labels from parquet, optionally subsample."""
    print(f"Loading data from {INPUT_PATH}...")
    table = pq.read_table(str(INPUT_PATH))
    n_total = len(table)
    print(f"  Total rows: {n_total:,}")

    # Extract columns
    l1_system = table.column("level1_system").to_pylist()
    l4_subtype = table.column("level4_subtype").to_pylist()

    # Build label mappings
    l0_names = sorted(set(l1_system))
    l1_names = sorted(set(l4_subtype))
    l0_to_idx = {n: i for i, n in enumerate(l0_names)}
    l1_to_idx = {n: i for i, n in enumerate(l1_names)}
    print(f"  L0 classes (system): {len(l0_names)}")
    print(f"  L1 classes (subtype): {len(l1_names)}")

    l0_labels = np.array([l0_to_idx[s] for s in l1_system], dtype=np.int64)
    l1_labels = np.array([l1_to_idx[s] for s in l4_subtype], dtype=np.int64)

    # Extract embeddings
    emb_column = table.column("embedding")
    embeddings = np.stack([np.array(emb_column[i].as_py(), dtype=np.float32) for i in range(n_total)])

    # Stratified subsample by L1 label if needed
    if max_samples and max_samples < n_total:
        print(f"  Stratified subsampling to {max_samples:,} rows...")
        rng = np.random.RandomState(seed)
        indices = []
        unique_l1 = np.unique(l1_labels)
        samples_per_class = max(1, max_samples // len(unique_l1))
        for label in unique_l1:
            class_indices = np.where(l1_labels == label)[0]
            n_take = min(samples_per_class, len(class_indices))
            chosen = rng.choice(class_indices, n_take, replace=False)
            indices.extend(chosen)
        # If we need more to reach max_samples, sample remainder
        if len(indices) < max_samples:
            remaining = list(set(range(n_total)) - set(indices))
            extra = rng.choice(remaining, max_samples - len(indices), replace=False)
            indices.extend(extra)
        indices = np.array(indices[:max_samples])
        rng.shuffle(indices)

        embeddings = embeddings[indices]
        l0_labels = l0_labels[indices]
        l1_labels = l1_labels[indices]
        print(f"  Subsampled: {len(embeddings):,} rows")

    # Train/val split (stratified by L1)
    rng = np.random.RandomState(seed)
    n = len(embeddings)
    perm = rng.permutation(n)
    n_val = int(n * val_ratio)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = CellEmbeddingDataset(
        torch.from_numpy(embeddings[train_idx]),
        torch.from_numpy(l0_labels[train_idx]),
        torch.from_numpy(l1_labels[train_idx]),
    )
    val_ds = CellEmbeddingDataset(
        torch.from_numpy(embeddings[val_idx]),
        torch.from_numpy(l0_labels[val_idx]),
        torch.from_numpy(l1_labels[val_idx]),
    )

    print(f"  Train: {len(train_ds):,}, Val: {len(val_ds):,}")
    return train_ds, val_ds, l0_names, l1_names


def contrastive_loss(anchor, positive, temperature=TEMPERATURE):
    """InfoNCE contrastive loss."""
    logits = anchor @ positive.T / temperature
    targets = torch.arange(len(anchor), device=anchor.device)
    return F.cross_entropy(logits, targets)


def margin_loss(embeddings, labels, margin=MARGIN):
    """In-batch margin ranking loss."""
    sims = embeddings @ embeddings.T
    same_class = labels.unsqueeze(1) == labels.unsqueeze(0)
    same_class.fill_diagonal_(False)
    diff_class = ~(labels.unsqueeze(1) == labels.unsqueeze(0))
    diff_class.fill_diagonal_(False)

    losses = []
    for i in range(len(embeddings)):
        pos_mask = same_class[i]
        neg_mask = diff_class[i]
        if not pos_mask.any() or not neg_mask.any():
            continue
        pos_sims = sims[i][pos_mask]
        neg_sims = sims[i][neg_mask]
        for pos_sim in pos_sims:
            loss = F.relu(neg_sims - pos_sim + margin).mean()
            losses.append(loss)

    if losses:
        return torch.stack(losses).mean()
    return torch.tensor(0.0, device=embeddings.device)


def sample_prefix_lengths(batch_size, device):
    """Sample prefix lengths per V5 design: P(j=1)=0.6, P(j=2)=0.4."""
    probs = torch.tensor(PREFIX_PROBS)
    lengths = torch.multinomial(probs.expand(batch_size, -1), num_samples=1).squeeze(-1)
    return (lengths + 1).to(device)  # 1-indexed


def create_block_dropout_mask(batch_size, prefix_lengths, device):
    """Zero out blocks beyond prefix length for prefix path."""
    mask = torch.zeros(batch_size, NUM_SCALES, device=device)
    for i in range(batch_size):
        j = prefix_lengths[i].item()
        mask[i, :j] = 1.0
    return mask


def create_full_dropout_mask(batch_size, device):
    """Block dropout for full path: keep_probs=[0.95, 0.7]."""
    mask = torch.ones(batch_size, NUM_SCALES, device=device)
    for block_idx, keep_prob in enumerate(BLOCK_KEEP_PROBS):
        drop = torch.rand(batch_size, device=device) > keep_prob
        mask[drop, block_idx] = 0.0
    return mask


def knn_accuracy(embeddings, labels, k=5):
    """k-NN accuracy: for each point, predict label from k nearest neighbors."""
    embeddings = F.normalize(embeddings.cpu(), dim=-1)
    labels = labels.cpu()
    sims = embeddings @ embeddings.T
    sims.fill_diagonal_(-float("inf"))
    _, top_k_idx = sims.topk(k, dim=1)
    neighbor_labels = labels[top_k_idx]  # (N, k)
    # Majority vote
    preds = torch.mode(neighbor_labels, dim=1).values
    return (preds == labels).float().mean().item()


def evaluate(model, val_ds, device, max_eval=2000):
    """Evaluate kNN accuracy on validation set."""
    model.eval()
    n = min(len(val_ds), max_eval)
    emb = val_ds.embeddings[:n].to(device)
    l0 = val_ds.l0_labels[:n].to(device)
    l1 = val_ds.l1_labels[:n].to(device)

    with torch.no_grad():
        result = model(emb)
        full_emb = result["full_embedding"]

    l0_acc = knn_accuracy(full_emb, l0)
    l1_acc = knn_accuracy(full_emb, l1)
    return {"l0_accuracy": l0_acc, "l1_accuracy": l1_acc}


def train(args):
    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Load data
    train_ds, val_ds, l0_names, l1_names = load_data(args.max_samples, args.val_ratio, args.seed)
    num_l0 = len(l0_names)
    num_l1 = len(l1_names)

    # Model
    model = FractalHeadV5(
        input_dim=INPUT_DIM,
        num_scales=NUM_SCALES,
        scale_dim=SCALE_DIM,
        num_l0_classes=num_l0,
        num_l1_classes=num_l1,
        num_heads=8,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"FractalHeadV5: {total_params:,} parameters")
    print(f"  scales={NUM_SCALES}, scale_dim={SCALE_DIM}, output_dim={NUM_SCALES * SCALE_DIM}")
    print(f"  L0 classes={num_l0}, L1 classes={num_l1}")

    # DataLoader
    def collate(batch):
        return {
            "anchor": torch.stack([b["anchor"] for b in batch]),
            "l0_pos": torch.stack([b["l0_pos"] for b in batch]),
            "l1_pos": torch.stack([b["l1_pos"] for b in batch]),
            "l0_label": torch.stack([b["l0_label"] for b in batch]),
            "l1_label": torch.stack([b["l1_label"] for b in batch]),
        }

    dataloader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, drop_last=True, num_workers=0,
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(dataloader))

    best_score = -float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            anchor = batch["anchor"].to(device)
            l0_pos = batch["l0_pos"].to(device)
            l1_pos = batch["l1_pos"].to(device)
            l0_labels = batch["l0_label"].to(device)
            l1_labels = batch["l1_label"].to(device)
            bs = anchor.shape[0]

            optimizer.zero_grad()

            # === FULL PATH ===
            full_dropout = create_full_dropout_mask(bs, device)
            anchor_full = model(anchor, block_dropout_mask=full_dropout)
            l1_pos_full = model(l1_pos)

            full_emb = anchor_full["full_embedding"]
            l1_pos_emb = l1_pos_full["full_embedding"]

            loss_full = (
                contrastive_loss(full_emb, l1_pos_emb)
                + MARGIN_WEIGHT * margin_loss(full_emb, l1_labels)
                + CLASS_WEIGHT * F.cross_entropy(model.classify_leaf(full_emb), l1_labels)
            )

            # === PREFIX PATH ===
            prefix_lengths = sample_prefix_lengths(bs, device)
            prefix_dropout = create_block_dropout_mask(bs, prefix_lengths, device)
            anchor_prefix = model(anchor, block_dropout_mask=prefix_dropout)
            l0_pos_prefix = model(l0_pos)

            mode_prefix_len = prefix_lengths.cpu().mode().values.item()
            prefix_emb = model.get_prefix_embedding(anchor_prefix["blocks"], mode_prefix_len)
            l0_pos_emb = l0_pos_prefix["full_embedding"]

            loss_prefix = (
                contrastive_loss(prefix_emb, l0_pos_emb)
                + MARGIN_WEIGHT * margin_loss(prefix_emb, l0_labels)
                + CLASS_WEIGHT * F.cross_entropy(model.classify_top(prefix_emb), l0_labels)
            )

            # === TOTAL ===
            loss = loss_full + PREFIX_WEIGHT * loss_prefix

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)

        # Evaluate
        val_score = evaluate(model, val_ds, device)
        score = val_score["l0_accuracy"] + val_score["l1_accuracy"]
        print(
            f"  Epoch {epoch + 1}: loss={avg_loss:.4f}, "
            f"L0={val_score['l0_accuracy']:.4f}, L1={val_score['l1_accuracy']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    # Save best model
    if best_state is not None:
        model.load_state_dict(best_state)
    save_dict = {
        "model_state_dict": model.state_dict(),
        "config": {
            "input_dim": INPUT_DIM,
            "num_scales": NUM_SCALES,
            "scale_dim": SCALE_DIM,
            "num_l0_classes": num_l0,
            "num_l1_classes": num_l1,
        },
        "l0_names": l0_names,
        "l1_names": l1_names,
    }
    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, OUTPUT_MODEL_PATH)
    print(f"\nSaved model to {OUTPUT_MODEL_PATH}")
    print(f"Best score: {best_score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train fractal adapter for cell embeddings")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
