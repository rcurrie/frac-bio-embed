#!/usr/bin/env python3
"""Generate fractal embeddings from a trained adapter.

Reads the full scimilarity_embeddings.parquet, transforms each 128d embedding
through the trained FractalHeadV5, and writes fractal_embeddings.parquet with
the same schema (128d fractal embedding + 4-level hierarchy labels).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "fractal-embeddings" / "moonshot-fractal-embeddings" / "src"))
from fractal_v5 import FractalHeadV5

INPUT_PATH = Path("data/scimilarity_embeddings.parquet")
OUTPUT_PATH = Path("data/fractal_embeddings.parquet")
MODEL_PATH = Path("data/fractal_adapter.pt")
EMBEDDING_DIM = 128


def embed(args):
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

    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = FractalHeadV5(
        input_dim=config["input_dim"],
        num_scales=config["num_scales"],
        scale_dim=config["scale_dim"],
        num_l0_classes=config["num_l0_classes"],
        num_l1_classes=config["num_l1_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Loaded: {config['num_scales']} scales x {config['scale_dim']}d = {config['num_scales'] * config['scale_dim']}d output")

    # Read input parquet metadata
    pf = pq.ParquetFile(str(args.input))
    n_total = pf.metadata.num_rows
    n_row_groups = pf.metadata.num_row_groups
    print(f"Input: {n_total:,} rows, {n_row_groups} row groups")

    # Process row group by row group to limit memory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    for rg_idx in tqdm(range(n_row_groups), desc="Row groups"):
        rg_table = pf.read_row_group(rg_idx)
        rg_size = len(rg_table)

        # Extract hierarchy columns (pass through)
        l1_sys = rg_table.column("level1_system")
        l2_lin = rg_table.column("level2_lineage")
        l3_typ = rg_table.column("level3_type")
        l4_sub = rg_table.column("level4_subtype")

        # Extract embeddings and transform in batches
        emb_column = rg_table.column("embedding")
        all_fractal = []

        for batch_start in range(0, rg_size, args.batch_size):
            batch_end = min(batch_start + args.batch_size, rg_size)
            batch_embs = np.stack([
                np.array(emb_column[i].as_py(), dtype=np.float32)
                for i in range(batch_start, batch_end)
            ])

            with torch.no_grad():
                inp = torch.from_numpy(batch_embs).to(device)
                result = model(inp)
                fractal = result["full_embedding"].cpu().numpy()
            all_fractal.append(fractal)

        fractal_embs = np.concatenate(all_fractal, axis=0)  # (rg_size, 128)

        # Build output table
        flat = fractal_embs.ravel().astype(np.float32)
        embedding_col = pa.FixedSizeListArray.from_arrays(
            pa.array(flat, type=pa.float32()), list_size=EMBEDDING_DIM
        )

        out_table = pa.table({
            "embedding": embedding_col,
            "level1_system": l1_sys,
            "level2_lineage": l2_lin,
            "level3_type": l3_typ,
            "level4_subtype": l4_sub,
        })

        if writer is None:
            writer = pq.ParquetWriter(str(output_path), out_table.schema, compression="snappy")
        writer.write_table(out_table)

    if writer is not None:
        writer.close()

    # Validate
    out_pf = pq.ParquetFile(str(output_path))
    print(f"\nOutput: {output_path}")
    print(f"  Rows: {out_pf.metadata.num_rows:,}")
    print(f"  Row groups: {out_pf.metadata.num_row_groups}")
    print(f"  File size: {output_path.stat().st_size / (1024**3):.2f} GB")

    sample = pq.read_table(str(output_path)).slice(0, 3)
    for i in range(3):
        emb = sample.column("embedding")[i].as_py()
        l1 = sample.column("level1_system")[i].as_py()
        l4 = sample.column("level4_subtype")[i].as_py()
        print(f"  [{i}] L1={l1}, L4={l4}, emb[:5]={emb[:5]}")


def main():
    parser = argparse.ArgumentParser(description="Generate fractal embeddings")
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--input", type=str, default=str(INPUT_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    embed(args)


if __name__ == "__main__":
    main()
