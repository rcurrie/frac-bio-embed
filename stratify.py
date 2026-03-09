#!/usr/bin/env python3
"""Pre-stratify the scimilarity embeddings parquet for balanced sequential reads.

Reorders rows via round-robin interleaving by level4_subtype (finest grain),
which automatically stratifies all coarser levels since each L4 maps to exactly
one L3, L2, L1. The output parquet can be read sequentially (first N rows) and
will have balanced representation across all hierarchy levels.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

INPUT_PATH = "data/scimilarity_embeddings.parquet"
OUTPUT_PATH = "data/stratified.parquet"
BATCH_SIZE = 500_000
EMBEDDING_DIM = 128


def stratify(args):
    input_path = args.input
    output_path = args.output
    seed = args.seed

    # Open parquet file
    if input_path.startswith("s3://"):
        import s3fs
        fs = s3fs.S3FileSystem()
        pf = pq.ParquetFile(input_path, filesystem=fs)
    else:
        pf = pq.ParquetFile(input_path)

    n_total = pf.metadata.num_rows
    n_row_groups = pf.metadata.num_row_groups
    print(f"Input: {n_total:,} rows, {n_row_groups} row groups")

    # Phase 1: Read all label columns (cheap, ~32MB)
    print("Reading label columns...")
    label_tables = [
        pf.read_row_group(i, columns=["level1_system", "level2_lineage", "level3_type", "level4_subtype"])
        for i in tqdm(range(n_row_groups), desc="Label row groups")
    ]
    label_table = pa.concat_tables(label_tables)

    l1_list = label_table.column("level1_system").to_pylist()
    l2_list = label_table.column("level2_lineage").to_pylist()
    l3_list = label_table.column("level3_type").to_pylist()
    l4_list = label_table.column("level4_subtype").to_pylist()

    # Phase 2: Build round-robin order by L4 subtype
    print("Building stratified order...")
    groups = defaultdict(list)
    for i, l4 in enumerate(l4_list):
        groups[l4].append(i)

    rng = np.random.RandomState(seed)
    for indices in groups.values():
        rng.shuffle(indices)

    # Sort groups by size descending for better interleaving
    sorted_groups = sorted(groups.values(), key=len, reverse=True)
    print(f"  {len(sorted_groups)} L4 subtypes, largest group: {len(sorted_groups[0]):,}, smallest: {len(sorted_groups[-1]):,}")

    # Round-robin across groups
    order = []
    iterators = [iter(g) for g in sorted_groups]
    while iterators:
        next_round = []
        for it in iterators:
            val = next(it, None)
            if val is not None:
                order.append(val)
                next_round.append(it)
        iterators = next_round
    order = np.array(order)
    assert len(order) == n_total, f"Order length {len(order)} != total {n_total}"

    # Phase 3: Read all embeddings
    print("Reading all embeddings...")
    emb_tables = [
        pf.read_row_group(i, columns=["embedding"])
        for i in tqdm(range(n_row_groups), desc="Embedding row groups")
    ]
    emb_table = pa.concat_tables(emb_tables)
    emb_column = emb_table.column("embedding")

    print("Extracting embeddings to numpy...")
    all_embeddings = np.stack([
        np.array(emb_column[i].as_py(), dtype=np.float32)
        for i in tqdm(range(n_total), desc="Rows", mininterval=5)
    ])

    # Phase 4: Write in new order
    print(f"Writing stratified parquet to {output_path}...")
    writer = None

    for batch_start in tqdm(range(0, n_total, BATCH_SIZE), desc="Writing batches"):
        batch_end = min(batch_start + BATCH_SIZE, n_total)
        batch_indices = order[batch_start:batch_end]

        batch_emb = all_embeddings[batch_indices]
        flat = batch_emb.ravel().astype(np.float32)
        embedding_col = pa.FixedSizeListArray.from_arrays(
            pa.array(flat, type=pa.float32()), list_size=EMBEDDING_DIM
        )

        table = pa.table({
            "embedding": embedding_col,
            "level1_system": pa.array([l1_list[i] for i in batch_indices], type=pa.utf8()).dictionary_encode(),
            "level2_lineage": pa.array([l2_list[i] for i in batch_indices], type=pa.utf8()).dictionary_encode(),
            "level3_type": pa.array([l3_list[i] for i in batch_indices], type=pa.utf8()).dictionary_encode(),
            "level4_subtype": pa.array([l4_list[i] for i in batch_indices], type=pa.utf8()).dictionary_encode(),
        })

        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
        writer.write_table(table)

    if writer is not None:
        writer.close()

    # Phase 5: Validate stratification
    print("\nValidation:")
    out_pf = pq.ParquetFile(output_path)
    print(f"  Output: {out_pf.metadata.num_rows:,} rows, {out_pf.metadata.num_row_groups} row groups")
    out_size = Path(output_path).stat().st_size / (1024**3)
    print(f"  File size: {out_size:.2f} GB")

    # Check L1 distribution in first vs last 10k rows
    out_table = pq.read_table(output_path)
    for label, name in [("level1_system", "L1 system"), ("level4_subtype", "L4 subtype")]:
        first = out_table.column(label).to_pylist()[:10_000]
        last = out_table.column(label).to_pylist()[-10_000:]
        first_dist = defaultdict(int)
        last_dist = defaultdict(int)
        for v in first:
            first_dist[v] += 1
        for v in last:
            last_dist[v] += 1

        all_keys = sorted(set(first_dist) | set(last_dist))
        print(f"\n  {name} distribution (first 10k vs last 10k):")
        for k in all_keys[:10]:  # show top 10
            f_pct = first_dist.get(k, 0) / 100
            l_pct = last_dist.get(k, 0) / 100
            print(f"    {k:<30s}  first: {f_pct:5.1f}%  last: {l_pct:5.1f}%")
        if len(all_keys) > 10:
            print(f"    ... and {len(all_keys) - 10} more classes")

    unique_l4_first = len(set(out_table.column("level4_subtype").to_pylist()[:10_000]))
    unique_l4_total = len(set(out_table.column("level4_subtype").to_pylist()))
    print(f"\n  L4 subtypes in first 10k: {unique_l4_first} / {unique_l4_total} total")


def main():
    parser = argparse.ArgumentParser(description="Pre-stratify embeddings parquet for balanced sequential reads")
    parser.add_argument("--input", type=str, default=INPUT_PATH)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    stratify(args)


if __name__ == "__main__":
    main()
