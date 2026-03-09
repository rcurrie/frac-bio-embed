#!/usr/bin/env python3
"""Train a fractal adapter to transform 128d SCimilarity embeddings into
hierarchy-aligned fractal embeddings.

Uses FractalHeadV5 from the fractal-embeddings submodule. Supports 2, 3, or 4
hierarchy levels via --num-levels:
  2: level1_system + level2_lineage
  3: level1_system + level2_lineage + level3_type
  4: level1_system + level2_lineage + level3_type + level4_subtype

The adapter learns to produce (num_levels x 64d) output where successive prefixes
capture increasingly fine-grained structure.

Training uses the same loss recipe as fractal_v5: contrastive + margin +
classification with prefix supervision and block dropout.
"""

import argparse
import io
import os
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import FractalHeadV5 from submodule
sys.path.insert(0, str(Path(__file__).parent / "fractal-embeddings" / "moonshot-fractal-embeddings" / "src"))
from fractal_v5 import FractalHeadV5

INPUT_PATH = Path("data/stratified.parquet")
OUTPUT_MODEL_PATH = Path("data/fractal_adapter.pt")

# Fractal architecture
INPUT_DIM = 128

# Column mapping: index -> parquet column name
LEVEL_COLUMNS = ["level1_system", "level2_lineage", "level3_type", "level4_subtype"]

# Loss hyperparameters
MARGIN_WEIGHT = 0.5
CLASS_WEIGHT = 1.0
TEMPERATURE = 0.07
MARGIN = 0.2
PREFIX_WEIGHT = 0.6


def get_prefix_probs(num_scales):
    """Prefix sampling probabilities. Heavier weight on coarse prefixes."""
    if num_scales == 2:
        return [0.6, 0.4]
    elif num_scales == 3:
        return [0.45, 0.35, 0.2]
    elif num_scales == 4:
        return [0.4, 0.3, 0.2, 0.1]
    raise ValueError(f"Unsupported num_scales={num_scales}")


def get_block_keep_probs(num_scales):
    """Block dropout probabilities. Earlier blocks kept more often."""
    if num_scales == 2:
        return [0.95, 0.7]
    elif num_scales == 3:
        return [0.95, 0.85, 0.7]
    elif num_scales == 4:
        return [0.95, 0.9, 0.8, 0.7]
    raise ValueError(f"Unsupported num_scales={num_scales}")


class CellEmbeddingDataset(Dataset):
    """Dataset of pre-computed cell embeddings with hierarchical labels.

    Supports contrastive sampling: each __getitem__ returns an anchor plus
    a positive embedding for each hierarchy level.
    """

    def __init__(self, embeddings, labels_per_level):
        self.embeddings = embeddings  # (N, 128) float32 tensor
        self.num_levels = len(labels_per_level)
        self.labels = labels_per_level  # list of (N,) int64 tensors

        # Build indices for contrastive sampling at each level
        self.level_indices = []
        for level_labels in labels_per_level:
            index = {}
            for i in range(len(level_labels)):
                label = level_labels[i].item()
                index.setdefault(label, []).append(i)
            self.level_indices.append(index)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        result = {"anchor": self.embeddings[idx]}

        for lvl in range(self.num_levels):
            label = self.labels[lvl][idx].item()
            indices = self.level_indices[lvl][label]
            pos_idx = random.choice(indices)
            while pos_idx == idx and len(indices) > 1:
                pos_idx = random.choice(indices)
            result[f"l{lvl}_pos"] = self.embeddings[pos_idx]
            result[f"l{lvl}_label"] = self.labels[lvl][idx]

        return result


def load_data(input_path, num_levels, max_samples, val_ratio, seed):
    """Load embeddings and labels from pre-stratified parquet.

    With stratified parquet, subsampling is just taking the first N rows.
    """
    print(f"Loading data from {input_path}...")
    if input_path.startswith("s3://"):
        import s3fs
        fs = s3fs.S3FileSystem()
        pf = pq.ParquetFile(input_path, filesystem=fs)
    else:
        pf = pq.ParquetFile(input_path)

    n_total = pf.metadata.num_rows
    n_row_groups = pf.metadata.num_row_groups
    print(f"  Total rows: {n_total:,} ({n_row_groups} row groups)")

    columns = LEVEL_COLUMNS[:num_levels]
    print(f"  Using {num_levels} levels: {columns}")

    # Determine how many rows to use
    n_use = min(max_samples, n_total) if max_samples else n_total

    # Determine which row groups cover the first n_use rows
    rg_indices = []
    rows_available = 0
    for i in range(n_row_groups):
        rg_indices.append(i)
        rows_available += pf.metadata.row_group(i).num_rows
        if rows_available >= n_use:
            break

    print(f"  Reading {len(rg_indices)} of {n_row_groups} row groups ({rows_available:,} rows)")

    # Read label columns
    label_tables = [pf.read_row_group(i, columns=columns) for i in rg_indices]
    label_table = pa.concat_tables(label_tables)

    # Build label mappings per level
    level_names = []
    all_labels = []
    for col_name in columns:
        col_values = label_table.column(col_name).to_pylist()[:n_use]
        names = sorted(set(col_values))
        name_to_idx = {n: i for i, n in enumerate(names)}
        labels = np.array([name_to_idx[v] for v in col_values], dtype=np.int64)
        level_names.append(names)
        all_labels.append(labels)
        print(f"  {col_name}: {len(names)} classes")

    # Read embedding column
    emb_tables = [pf.read_row_group(i, columns=["embedding"]) for i in rg_indices]
    emb_table = pa.concat_tables(emb_tables)
    emb_column = emb_table.column("embedding")

    embeddings = np.stack([
        np.array(emb_column[i].as_py(), dtype=np.float32)
        for i in range(n_use)
    ])

    # Truncate labels to match
    all_labels = [labels[:n_use] for labels in all_labels]

    # Train/val split
    rng = np.random.RandomState(seed)
    n = len(embeddings)
    perm = rng.permutation(n)
    n_val = int(n * val_ratio)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = CellEmbeddingDataset(
        torch.from_numpy(embeddings[train_idx]),
        [torch.from_numpy(lab[train_idx]) for lab in all_labels],
    )
    val_ds = CellEmbeddingDataset(
        torch.from_numpy(embeddings[val_idx]),
        [torch.from_numpy(lab[val_idx]) for lab in all_labels],
    )

    print(f"  Train: {len(train_ds):,}, Val: {len(val_ds):,}")
    return train_ds, val_ds, level_names


def contrastive_loss(anchor, positive, temperature=TEMPERATURE):
    """InfoNCE contrastive loss."""
    logits = anchor @ positive.T / temperature
    targets = torch.arange(len(anchor), device=anchor.device)
    return F.cross_entropy(logits, targets)


def margin_loss(embeddings, labels, margin=MARGIN):
    """In-batch margin ranking loss (vectorized).

    For each sample, computes max(0, neg_sim - pos_sim + margin) averaged over
    all (positive, negative) pairs. Uses masked operations to avoid per-element
    Python loops that cause memory bloat from accumulated computation graphs.
    """
    sims = embeddings @ embeddings.T
    same_class = labels.unsqueeze(1) == labels.unsqueeze(0)
    same_class.fill_diagonal_(False)
    diff_class = ~(labels.unsqueeze(1) == labels.unsqueeze(0))
    diff_class.fill_diagonal_(False)

    # For each row: mean of positive sims, mean of negative sims
    # Then apply margin: relu(mean_neg - mean_pos + margin)
    pos_count = same_class.float().sum(dim=1)
    neg_count = diff_class.float().sum(dim=1)

    # Mask out rows with no positives or no negatives
    valid = (pos_count > 0) & (neg_count > 0)
    if not valid.any():
        return torch.tensor(0.0, device=embeddings.device)

    # Mean positive/negative similarity per row
    pos_sims = (sims * same_class.float()).sum(dim=1) / pos_count.clamp(min=1)
    neg_sims = (sims * diff_class.float()).sum(dim=1) / neg_count.clamp(min=1)

    losses = F.relu(neg_sims[valid] - pos_sims[valid] + margin)
    return losses.mean()


def sample_prefix_lengths(batch_size, num_scales, prefix_probs, device):
    """Sample prefix lengths per V5 design."""
    probs = torch.tensor(prefix_probs)
    lengths = torch.multinomial(probs.expand(batch_size, -1), num_samples=1).squeeze(-1)
    return (lengths + 1).to(device)  # 1-indexed


def create_block_dropout_mask(batch_size, prefix_lengths, num_scales, device):
    """Zero out blocks beyond prefix length for prefix path."""
    mask = torch.zeros(batch_size, num_scales, device=device)
    for i in range(batch_size):
        j = prefix_lengths[i].item()
        mask[i, :j] = 1.0
    return mask


def create_full_dropout_mask(batch_size, num_scales, block_keep_probs, device):
    """Block dropout for full path."""
    mask = torch.ones(batch_size, num_scales, device=device)
    for block_idx, keep_prob in enumerate(block_keep_probs):
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


def evaluate(model, val_ds, device, num_levels, max_eval=2000):
    """Evaluate kNN accuracy on validation set at each hierarchy level."""
    model.eval()
    n = min(len(val_ds), max_eval)
    emb = val_ds.embeddings[:n].to(device)

    with torch.no_grad():
        result = model(emb)
        full_emb = result["full_embedding"]

    metrics = {}
    for lvl in range(num_levels):
        labels = val_ds.labels[lvl][:n].to(device)
        metrics[f"l{lvl}_accuracy"] = knn_accuracy(full_emb, labels)
    return metrics


def train(args):
    num_levels = args.num_levels
    num_scales = num_levels
    scale_dim = args.scale_dim

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

    # Dynamic hyperparameters
    prefix_probs = get_prefix_probs(num_scales)
    block_keep_probs = get_block_keep_probs(num_scales)
    print(f"Levels: {num_levels}, prefix_probs={prefix_probs}, block_keep={block_keep_probs}")

    # Resolve input/output paths
    if args.s3_base:
        s3_base = args.s3_base.rstrip("/")
        input_path = f"{s3_base}/stratified.parquet"
        output_model_path = args.output or f"{s3_base}/fractal_adapter_{num_levels}L.pt"
    else:
        input_path = str(INPUT_PATH)
        output_model_path = args.output or str(OUTPUT_MODEL_PATH).replace(".pt", f"_{num_levels}L.pt")

    # Load data
    train_ds, val_ds, level_names = load_data(input_path, num_levels, args.max_samples, args.val_ratio, args.seed)
    num_top_classes = len(level_names[0])
    num_leaf_classes = len(level_names[-1])

    # Model
    model = FractalHeadV5(
        input_dim=INPUT_DIM,
        num_scales=num_scales,
        scale_dim=scale_dim,
        num_l0_classes=num_top_classes,
        num_l1_classes=num_leaf_classes,
        num_heads=8,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"FractalHeadV5: {total_params:,} parameters")
    print(f"  scales={num_scales}, scale_dim={scale_dim}, output_dim={num_scales * scale_dim}")
    print(f"  Top classes={num_top_classes}, Leaf classes={num_leaf_classes}")

    # DataLoader
    def collate(batch):
        result = {"anchor": torch.stack([b["anchor"] for b in batch])}
        for lvl in range(num_levels):
            result[f"l{lvl}_pos"] = torch.stack([b[f"l{lvl}_pos"] for b in batch])
            result[f"l{lvl}_label"] = torch.stack([b[f"l{lvl}_label"] for b in batch])
        return result

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

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}",
                    disable=not sys.stderr.isatty())
        for batch in pbar:
            anchor = batch["anchor"].to(device)
            bs = anchor.shape[0]

            optimizer.zero_grad()

            # === FULL PATH (leaf level) ===
            leaf_lvl = num_levels - 1
            leaf_pos = batch[f"l{leaf_lvl}_pos"].to(device)
            leaf_labels = batch[f"l{leaf_lvl}_label"].to(device)

            full_dropout = create_full_dropout_mask(bs, num_scales, block_keep_probs, device)
            anchor_full = model(anchor, block_dropout_mask=full_dropout)
            leaf_pos_full = model(leaf_pos)

            full_emb = anchor_full["full_embedding"]
            leaf_pos_emb = leaf_pos_full["full_embedding"]

            loss_full = (
                contrastive_loss(full_emb, leaf_pos_emb)
                + MARGIN_WEIGHT * margin_loss(full_emb, leaf_labels)
                + CLASS_WEIGHT * F.cross_entropy(model.classify_leaf(full_emb), leaf_labels)
            )

            # === PREFIX PATH ===
            prefix_lengths = sample_prefix_lengths(bs, num_scales, prefix_probs, device)
            prefix_dropout = create_block_dropout_mask(bs, prefix_lengths, num_scales, device)
            anchor_prefix = model(anchor, block_dropout_mask=prefix_dropout)

            mode_j = prefix_lengths.cpu().mode().values.item()
            level_idx = mode_j - 1  # 0-indexed level for this prefix

            level_pos = batch[f"l{level_idx}_pos"].to(device)
            level_labels = batch[f"l{level_idx}_label"].to(device)

            level_pos_result = model(level_pos)
            prefix_emb = model.get_prefix_embedding(anchor_prefix["blocks"], mode_j)
            level_pos_emb = level_pos_result["full_embedding"]

            # Contrastive + margin always apply
            loss_prefix = (
                contrastive_loss(prefix_emb, level_pos_emb)
                + MARGIN_WEIGHT * margin_loss(prefix_emb, level_labels)
            )

            # Classification only for top (j=1) and leaf (j=num_scales)
            if mode_j == 1:
                loss_prefix += CLASS_WEIGHT * F.cross_entropy(model.classify_top(prefix_emb), level_labels)
            elif mode_j == num_scales:
                loss_prefix += CLASS_WEIGHT * F.cross_entropy(model.classify_leaf(prefix_emb), level_labels)

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
        val_score = evaluate(model, val_ds, device, num_levels)
        score = sum(val_score[f"l{i}_accuracy"] for i in range(num_levels))
        metrics_str = ", ".join(f"L{i}={val_score[f'l{i}_accuracy']:.4f}" for i in range(num_levels))
        print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}, {metrics_str}")

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
            "num_scales": num_scales,
            "scale_dim": scale_dim,
            "num_l0_classes": num_top_classes,
            "num_l1_classes": num_leaf_classes,
            "num_levels": num_levels,
        },
        "level_names": level_names,
    }
    if output_model_path.startswith("s3://"):
        import boto3
        from botocore.config import Config
        buf = io.BytesIO()
        torch.save(save_dict, buf)
        buf.seek(0)
        parts = output_model_path[5:].split("/", 1)
        bucket, key = parts[0], parts[1]
        s3 = boto3.client("s3", config=Config(request_checksum_calculation="when_required"))
        s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    else:
        Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, output_model_path)
    print(f"\nSaved model to {output_model_path}")
    print(f"Best score: {best_score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train fractal adapter for cell embeddings")
    parser.add_argument("--num-levels", type=int, default=2, choices=[2, 3, 4],
                        help="Number of hierarchy levels (2=system+lineage, 3=+type, 4=+subtype)")
    parser.add_argument("--scale-dim", type=int, default=64,
                        help="Dimension per scale block (output_dim = num_levels * scale_dim)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default=None,
                        help="Output model checkpoint path (default: data/fractal_adapter_{N}L.pt)")
    parser.add_argument("--s3-base", type=str, default=os.environ.get("S3_BASE"),
                        help="S3 base path (e.g. s3://bucket/path). Falls back to S3_BASE env var.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
