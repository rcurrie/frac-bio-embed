#!/usr/bin/env python3
"""Evaluate fractal embeddings vs original SCimilarity embeddings.

Compares kNN accuracy at each prefix length, computes steerability metric,
and reports silhouette scores. Supports multi-level checkpoints (2, 3, or 4 levels).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent / "fractal-embeddings" / "moonshot-fractal-embeddings" / "src"))
from fractal_v5 import FractalHeadV5

INPUT_PATH = Path("data/stratified.parquet")
MODEL_PATH = Path("data/fractal_adapter.pt")

LEVEL_COLUMNS = ["level1_system", "level2_lineage", "level3_type", "level4_subtype"]
LEVEL_SHORT = ["system", "lineage", "type", "subtype"]


def knn_accuracy(embeddings, labels, k=5):
    """k-NN accuracy using cosine similarity."""
    emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)

    correct = 0
    for i in range(len(emb)):
        top_k = np.argsort(-sims[i])[:k]
        neighbor_labels = labels[top_k]
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        pred = unique[np.argmax(counts)]
        if pred == labels[i]:
            correct += 1
    return correct / len(emb)


def evaluate(args):
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

    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    num_levels = config.get("num_levels", 2)
    num_scales = config["num_scales"]
    scale_dim = config["scale_dim"]

    # Handle both old (l0_names/l1_names) and new (level_names) checkpoint formats
    if "level_names" in checkpoint:
        level_names = checkpoint["level_names"]
    else:
        level_names = [checkpoint["l0_names"], checkpoint["l1_names"]]

    model = FractalHeadV5(
        input_dim=config["input_dim"],
        num_scales=num_scales,
        scale_dim=scale_dim,
        num_l0_classes=config["num_l0_classes"],
        num_l1_classes=config["num_l1_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  {num_levels} levels, {num_scales} scales x {scale_dim}d = {num_scales * scale_dim}d output")

    # Determine which columns this model was trained on
    columns = LEVEL_COLUMNS[:num_levels]

    # Load data
    print(f"Loading data from {args.input}...")
    table = pq.read_table(str(args.input))
    n_total = len(table)

    # Build labels per level
    all_labels = []
    for lvl, col_name in enumerate(columns):
        col_values = table.column(col_name).to_pylist()
        name_to_idx = {n: i for i, n in enumerate(level_names[lvl])}
        labels = np.array([name_to_idx[v] for v in col_values], dtype=np.int64)
        all_labels.append(labels)

    emb_column = table.column("embedding")
    embeddings = np.stack([np.array(emb_column[i].as_py(), dtype=np.float32) for i in range(n_total)])

    # Subsample
    rng = np.random.RandomState(args.seed)
    n = min(args.num_samples, n_total)
    indices = rng.choice(n_total, n, replace=False)
    embeddings = embeddings[indices]
    all_labels = [lab[indices] for lab in all_labels]
    print(f"  Evaluating on {n:,} samples")

    # === Baseline: original embeddings ===
    print(f"\n--- Original Embeddings ({INPUT_PATH.name}, 128d) ---")
    for lvl in range(num_levels):
        acc = knn_accuracy(embeddings, all_labels[lvl])
        print(f"  kNN L{lvl} accuracy ({LEVEL_SHORT[lvl]}): {acc:.4f}")

    orig_normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    orig_sil = silhouette_score(orig_normed, all_labels[0], metric="cosine", sample_size=min(n, 5000))
    print(f"  Silhouette (L0, cosine):   {orig_sil:.4f}")

    # === Fractal embeddings ===
    print(f"\n--- Fractal Embeddings ({num_scales} scales x {scale_dim}d) ---")
    with torch.no_grad():
        inp = torch.from_numpy(embeddings).to(device)
        result = model(inp)
        fractal_full = result["full_embedding"].cpu().numpy()
        blocks = result["blocks"]

    # Full embedding
    print(f"  Full ({num_scales * scale_dim}d):")
    full_accs = {}
    for lvl in range(num_levels):
        acc = knn_accuracy(fractal_full, all_labels[lvl])
        full_accs[lvl] = acc
        print(f"    kNN L{lvl} accuracy ({LEVEL_SHORT[lvl]}): {acc:.4f}")

    full_sil = silhouette_score(fractal_full, all_labels[0], metric="cosine", sample_size=min(n, 5000))
    print(f"    Silhouette (L0, cosine):   {full_sil:.4f}")

    # Prefix embeddings at each scale
    prefix_accs = {}
    for j in range(1, num_scales + 1):
        with torch.no_grad():
            prefix_emb = model.get_prefix_embedding(blocks, prefix_len=j).cpu().numpy()
        dim = j * scale_dim
        print(f"  Prefix j={j} ({dim}d):")
        prefix_accs[j] = {}
        for lvl in range(num_levels):
            acc = knn_accuracy(prefix_emb, all_labels[lvl])
            prefix_accs[j][lvl] = acc
            print(f"    kNN L{lvl} accuracy ({LEVEL_SHORT[lvl]}): {acc:.4f}")

    # === Steerability ===
    # For the top level: compare prefix-1 vs full
    print(f"\n--- Steerability ---")
    top_prefix_acc = prefix_accs[1][0]
    top_full_acc = full_accs[0]
    leaf_full_acc = full_accs[num_levels - 1]
    leaf_prefix_acc = prefix_accs[1][num_levels - 1]
    steerability = (top_prefix_acc - top_full_acc) + (leaf_full_acc - leaf_prefix_acc)
    print(f"  S = (L0@prefix1 - L0@full) + (Leaf@full - Leaf@prefix1)")
    print(f"  S = ({top_prefix_acc:.4f} - {top_full_acc:.4f}) + ({leaf_full_acc:.4f} - {leaf_prefix_acc:.4f})")
    print(f"  S = {steerability:.4f}")
    if steerability > 0:
        print("  Positive steerability: fractal structure is working!")
    else:
        print("  Non-positive steerability: prefix not yet specializing for coarse labels.")

    # === Summary table ===
    header_levels = "  ".join(f"L{i} Acc" for i in range(num_levels))
    print(f"\n{'='*70}")
    print(f"{'Embedding':<22} {'Dims':<6} {header_levels}  {'Sil(L0)':<8}")
    print(f"{'-'*70}")

    # Original
    orig_accs_str = "  ".join(f"{knn_accuracy(embeddings, all_labels[i]):<8.4f}" for i in range(num_levels))
    print(f"{'Original':<22} {'128':<6} {orig_accs_str}  {orig_sil:<8.4f}")

    # Full fractal
    full_accs_str = "  ".join(f"{full_accs[i]:<8.4f}" for i in range(num_levels))
    print(f"{'Fractal (full)':<22} {str(num_scales * scale_dim):<6} {full_accs_str}  {full_sil:<8.4f}")

    # Each prefix
    for j in range(1, num_scales + 1):
        dim = j * scale_dim
        pref_accs_str = "  ".join(f"{prefix_accs[j][i]:<8.4f}" for i in range(num_levels))
        print(f"{f'Fractal (prefix {j})':<22} {str(dim):<6} {pref_accs_str}")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fractal vs original embeddings")
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH),
                        help="Path to trained model checkpoint")
    parser.add_argument("--input", type=str, default=str(INPUT_PATH))
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
