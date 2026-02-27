#!/usr/bin/env python3
"""Evaluate fractal embeddings vs original SCimilarity embeddings.

Compares kNN accuracy at each prefix length, computes steerability metric,
and reports silhouette scores. This tests whether the fractal adapter learned
to specialize the first 64d prefix for coarse system-level classification.
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

INPUT_PATH = Path("data/scimilarity_embeddings.parquet")
MODEL_PATH = Path("data/fractal_adapter.pt")


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
    l0_names = checkpoint["l0_names"]
    l1_names = checkpoint["l1_names"]

    model = FractalHeadV5(
        input_dim=config["input_dim"],
        num_scales=config["num_scales"],
        scale_dim=config["scale_dim"],
        num_l0_classes=config["num_l0_classes"],
        num_l1_classes=config["num_l1_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load data (stratified sample)
    print(f"Loading data from {args.input}...")
    table = pq.read_table(str(args.input))
    n_total = len(table)

    l1_system = table.column("level1_system").to_pylist()
    l4_subtype = table.column("level4_subtype").to_pylist()

    l0_to_idx = {n: i for i, n in enumerate(l0_names)}
    l1_to_idx = {n: i for i, n in enumerate(l1_names)}
    l0_labels = np.array([l0_to_idx[s] for s in l1_system], dtype=np.int64)
    l1_labels = np.array([l1_to_idx[s] for s in l4_subtype], dtype=np.int64)

    emb_column = table.column("embedding")
    embeddings = np.stack([np.array(emb_column[i].as_py(), dtype=np.float32) for i in range(n_total)])

    # Subsample
    rng = np.random.RandomState(args.seed)
    n = min(args.num_samples, n_total)
    indices = rng.choice(n_total, n, replace=False)
    embeddings = embeddings[indices]
    l0_labels = l0_labels[indices]
    l1_labels = l1_labels[indices]
    print(f"  Evaluating on {n:,} samples")

    # === Baseline: original embeddings ===
    print("\n--- Original Embeddings (128d) ---")
    orig_l0_acc = knn_accuracy(embeddings, l0_labels)
    orig_l1_acc = knn_accuracy(embeddings, l1_labels)
    print(f"  kNN L0 accuracy (system):  {orig_l0_acc:.4f}")
    print(f"  kNN L1 accuracy (subtype): {orig_l1_acc:.4f}")

    orig_normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    orig_sil = silhouette_score(orig_normed, l0_labels, metric="cosine", sample_size=min(n, 5000))
    print(f"  Silhouette (L0, cosine):   {orig_sil:.4f}")

    # === Fractal embeddings ===
    print("\n--- Fractal Embeddings ---")
    with torch.no_grad():
        inp = torch.from_numpy(embeddings).to(device)
        result = model(inp)
        fractal_full = result["full_embedding"].cpu().numpy()  # (N, 128)
        blocks = result["blocks"]

        # Prefix: first scale only (64d)
        prefix_emb = model.get_prefix_embedding(blocks, prefix_len=1).cpu().numpy()

    # Full (128d)
    full_l0_acc = knn_accuracy(fractal_full, l0_labels)
    full_l1_acc = knn_accuracy(fractal_full, l1_labels)
    print(f"  Full (128d) kNN L0 accuracy:  {full_l0_acc:.4f}")
    print(f"  Full (128d) kNN L1 accuracy:  {full_l1_acc:.4f}")

    full_sil = silhouette_score(fractal_full, l0_labels, metric="cosine", sample_size=min(n, 5000))
    print(f"  Full (128d) Silhouette (L0):  {full_sil:.4f}")

    # Prefix (64d)
    prefix_l0_acc = knn_accuracy(prefix_emb, l0_labels)
    prefix_l1_acc = knn_accuracy(prefix_emb, l1_labels)
    print(f"  Prefix (64d) kNN L0 accuracy: {prefix_l0_acc:.4f}")
    print(f"  Prefix (64d) kNN L1 accuracy: {prefix_l1_acc:.4f}")

    prefix_normed = prefix_emb / np.linalg.norm(prefix_emb, axis=1, keepdims=True)
    prefix_sil = silhouette_score(prefix_normed, l0_labels, metric="cosine", sample_size=min(n, 5000))
    print(f"  Prefix (64d) Silhouette (L0): {prefix_sil:.4f}")

    # === Steerability ===
    steerability = (prefix_l0_acc - full_l0_acc) + (full_l1_acc - prefix_l1_acc)
    print(f"\n--- Steerability ---")
    print(f"  S = (L0@64d - L0@128d) + (L1@128d - L1@64d)")
    print(f"  S = ({prefix_l0_acc:.4f} - {full_l0_acc:.4f}) + ({full_l1_acc:.4f} - {prefix_l1_acc:.4f})")
    print(f"  S = {steerability:.4f}")
    if steerability > 0:
        print("  Positive steerability: fractal structure is working!")
    else:
        print("  Non-positive steerability: prefix not yet specializing for coarse labels.")

    # === Summary table ===
    print(f"\n{'='*60}")
    print(f"{'Embedding':<20} {'Dims':<6} {'L0 Acc':<10} {'L1 Acc':<10} {'Sil (L0)':<10}")
    print(f"{'-'*60}")
    print(f"{'Original':<20} {'128':<6} {orig_l0_acc:<10.4f} {orig_l1_acc:<10.4f} {orig_sil:<10.4f}")
    print(f"{'Fractal (full)':<20} {'128':<6} {full_l0_acc:<10.4f} {full_l1_acc:<10.4f} {full_sil:<10.4f}")
    print(f"{'Fractal (prefix)':<20} {'64':<6} {prefix_l0_acc:<10.4f} {prefix_l1_acc:<10.4f} {prefix_sil:<10.4f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fractal vs original embeddings")
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--input", type=str, default=str(INPUT_PATH))
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
