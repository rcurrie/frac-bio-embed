#!/usr/bin/env python3
"""Export SCimilarity embeddings with 4-level hierarchical cell ontology annotations.

Reads ~7.9M cell embeddings from the SCimilarity annotation kNN index and
maps their cell type labels to a 4-level hierarchy (System -> Lineage -> Type -> Subtype)
derived from the Cell Ontology DAG. Outputs a single Parquet file.
"""

import csv
from pathlib import Path

import hnswlib
import networkx as nx
import numpy as np
import obonet
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

MODEL_PATH = Path("data/scimilarity/model_v1.1")
KNN_INDEX_PATH = MODEL_PATH / "annotation" / "labelled_kNN.bin"
REFERENCE_LABELS_PATH = MODEL_PATH / "annotation" / "reference_labels.tsv"
LABEL_INTS_PATH = MODEL_PATH / "label_ints.csv"
CL_OBO_URL = "http://purl.obolibrary.org/obo/cl/cl-basic.obo"
OUTPUT_PATH = Path("data/scimilarity_embeddings.parquet")
EMBEDDING_DIM = 128
BATCH_SIZE = 500_000

# Level 1 (System) anchors: broad biological system categories.
# These are tried first; Stem/Progenitor is a fallback (see LEVEL1_FALLBACK).
LEVEL1_ANCHORS = [
    ("CL:0000988", "Immune"),
    ("CL:0000066", "Epithelial"),
    ("CL:0002319", "Neural"),
    ("CL:0000115", "Endothelial"),
    ("CL:0002320", "Stromal"),
    ("CL:0000187", "Muscle"),
]

# Stem/Progenitor is only assigned when NO primary Level 1 anchor is reachable.
# Many differentiated cells have short paths through progenitor->stem cell,
# so this must not compete on distance with primary anchors.
LEVEL1_FALLBACK = ("CL:0000034", "Stem/Progenitor")

# Level 2 (Lineage) anchors: subdivisions within systems.
LEVEL2_ANCHORS = [
    ("CL:0000542", "Lymphoid"),
    ("CL:0000766", "Myeloid"),
    ("CL:0001065", "Innate Lymphoid"),
    ("CL:0000540", "Neuron"),
    ("CL:0000125", "Glial"),
    ("CL:0000057", "Fibroblast"),
    ("CL:0000499", "Stromal"),
    ("CL:0000136", "Adipocyte"),
    ("CL:0000192", "Smooth Muscle"),
    ("CL:0000746", "Cardiac Muscle"),
    ("CL:0000188", "Skeletal Muscle"),
]

# Manual name -> CL ID for terms not in current ontology.
MANUAL_NAME_TO_CL = {
    "native cell": "CL:0000003",
    "animal cell": "CL:0000548",
}

# Manual hierarchy overrides for terms that can't be cleanly resolved via DAG.
# These are either obsolete terms, or cells whose ontology placement is ambiguous.
MANUAL_HIERARCHY = {
    "native cell": {
        "level1": "Other",
        "level2": "Other",
        "level3": "native cell",
        "level4": "native cell",
    },
    "animal cell": {
        "level1": "Other",
        "level2": "Other",
        "level3": "animal cell",
        "level4": "animal cell",
    },
}


def load_ontology() -> nx.MultiDiGraph:
    """Load the Cell Ontology from OBO Foundry."""
    print("Loading Cell Ontology from OBO Foundry...")
    graph = obonet.read_obo(CL_OBO_URL)
    print(f"  Loaded {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
    return graph


def build_name_to_cl_id(
    graph: nx.MultiDiGraph, label_names: list[str]
) -> dict[str, str]:
    """Map cell type names to CL IDs using ontology names and synonyms."""
    # Build reverse lookups
    name_to_id: dict[str, str] = {}
    synonym_to_id: dict[str, str] = {}
    for node_id, data in graph.nodes(data=True):
        if "name" in data:
            name_to_id[data["name"]] = node_id
        for syn in data.get("synonym", []):
            if '"' in syn:
                syn_text = syn.split('"')[1]
                synonym_to_id[syn_text.lower()] = node_id

    mapping: dict[str, str] = {}
    for name in label_names:
        # Try direct name match
        if name in name_to_id:
            mapping[name] = name_to_id[name]
            continue
        # Try stripping ", human" suffix
        stripped = name.replace(", human", "").strip()
        if stripped in name_to_id:
            mapping[name] = name_to_id[stripped]
            continue
        # Try synonym match (case-insensitive)
        if name.lower() in synonym_to_id:
            mapping[name] = synonym_to_id[name.lower()]
            continue
        if stripped.lower() in synonym_to_id:
            mapping[name] = synonym_to_id[stripped.lower()]
            continue
        # Try manual override
        if name in MANUAL_NAME_TO_CL:
            mapping[name] = MANUAL_NAME_TO_CL[name]
            continue
        mapping[name] = None

    return mapping


def find_nearest_anchor(
    cl_id: str,
    anchors: list[tuple[str, str]],
    graph: nx.MultiDiGraph,
) -> tuple[str | None, str | None]:
    """Find the nearest anchor node, with priority-based tie-breaking.

    Returns (label, anchor_cl_id) or (None, None) if no anchor found.
    Anchors earlier in the list win ties.
    """
    # In obonet, edges go child->parent (is_a), so nx.descendants = ontology ancestors
    try:
        ancestors = nx.descendants(graph, cl_id)
    except nx.NetworkXError:
        return None, None

    # Also include self (a node can be its own anchor)
    ancestors_plus_self = ancestors | {cl_id}

    best_label = None
    best_anchor = None
    best_dist = float("inf")
    best_priority = float("inf")

    for priority, (anchor_id, label) in enumerate(anchors):
        if anchor_id not in ancestors_plus_self:
            continue
        if anchor_id == cl_id:
            dist = 0
        else:
            try:
                dist = nx.shortest_path_length(graph, cl_id, anchor_id)
            except nx.NetworkXNoPath:
                continue

        # Prefer shorter distance; on tie, prefer higher priority (lower index)
        if dist < best_dist or (dist == best_dist and priority < best_priority):
            best_dist = dist
            best_label = label
            best_anchor = anchor_id
            best_priority = priority

    return best_label, best_anchor


def compute_level3(
    cl_id: str, level2_anchor_id: str | None, graph: nx.MultiDiGraph
) -> str:
    """Compute Level 3 (Type) as the parent of the leaf on the path to Level 2 anchor."""
    if level2_anchor_id is None or cl_id == level2_anchor_id:
        return graph.nodes[cl_id].get("name", cl_id)
    try:
        path = nx.shortest_path(graph, cl_id, level2_anchor_id)
        # path = [leaf, ..., anchor]
        if len(path) >= 3:
            return graph.nodes[path[1]].get("name", path[1])
        return graph.nodes[cl_id].get("name", cl_id)
    except nx.NetworkXNoPath:
        return graph.nodes[cl_id].get("name", cl_id)


def build_hierarchy_map(
    graph: nx.MultiDiGraph, name_to_cl: dict[str, str]
) -> dict[str, dict[str, str]]:
    """Build 4-level hierarchy for each unique cell type name."""
    hierarchy = {}

    for name, cl_id in name_to_cl.items():
        # Check manual overrides first
        if name in MANUAL_HIERARCHY:
            hierarchy[name] = MANUAL_HIERARCHY[name]
            continue

        if cl_id is None or cl_id not in graph:
            hierarchy[name] = {
                "level1": "Other",
                "level2": "Other",
                "level3": name,
                "level4": name,
            }
            continue

        # Two-pass Level 1: try primary anchors first, fall back to Stem/Progenitor
        l1_label, _ = find_nearest_anchor(cl_id, LEVEL1_ANCHORS, graph)
        if l1_label is None:
            fb_label, _ = find_nearest_anchor(
                cl_id, [LEVEL1_FALLBACK], graph
            )
            l1_label = fb_label or "Other"

        l2_label, l2_anchor = find_nearest_anchor(cl_id, LEVEL2_ANCHORS, graph)
        l3_name = compute_level3(cl_id, l2_anchor, graph)

        hierarchy[name] = {
            "level1": l1_label,
            "level2": l2_label or l1_label,
            "level3": l3_name,
            "level4": name,
        }

    return hierarchy


def load_reference_labels() -> list[str]:
    """Load cell type labels from reference_labels.tsv."""
    print(f"Loading reference labels from {REFERENCE_LABELS_PATH}...")
    with open(REFERENCE_LABELS_PATH) as f:
        labels = [line.strip() for line in f]
    print(f"  Loaded {len(labels):,} labels, {len(set(labels))} unique types")
    return labels


def load_unique_label_names() -> list[str]:
    """Load the 203 unique cell type names from label_ints.csv."""
    with open(LABEL_INTS_PATH) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        return [row[1] for row in reader]


def main():
    # Phase A: Build ontology hierarchy for 203 unique types
    graph = load_ontology()
    unique_names = load_unique_label_names()
    print(f"  {len(unique_names)} unique cell type names from label_ints.csv")

    name_to_cl = build_name_to_cl_id(graph, unique_names)
    unmapped = [n for n, cl in name_to_cl.items() if cl is None]
    if unmapped:
        print(f"ERROR: {len(unmapped)} unmapped cell type names:")
        for n in unmapped:
            print(f"  {n!r}")
        raise SystemExit(1)
    print("  All cell type names mapped to CL IDs")

    hierarchy_map = build_hierarchy_map(graph, name_to_cl)

    # Print hierarchy summary
    from collections import Counter

    l1_counts = Counter(h["level1"] for h in hierarchy_map.values())
    l2_counts = Counter(h["level2"] for h in hierarchy_map.values())
    print(f"\nHierarchy summary ({len(hierarchy_map)} types):")
    print(f"  Level 1 (System):  {len(l1_counts)} categories: {dict(l1_counts)}")
    print(f"  Level 2 (Lineage): {len(l2_counts)} categories")
    for l2, count in sorted(l2_counts.items(), key=lambda x: -x[1]):
        print(f"    {l2}: {count}")

    # Phase B: Load reference labels
    labels = load_reference_labels()
    num_cells = len(labels)

    # Verify all labels have hierarchy entries
    missing = set(labels) - set(hierarchy_map)
    if missing:
        print(f"ERROR: {len(missing)} labels not in hierarchy map: {missing}")
        raise SystemExit(1)

    # Phase C+D: Read embeddings from kNN index and write Parquet
    print(f"\nLoading kNN index from {KNN_INDEX_PATH}...")
    knn = hnswlib.Index(space="l2", dim=EMBEDDING_DIM)
    knn.load_index(str(KNN_INDEX_PATH))
    knn_count = knn.get_current_count()
    print(f"  kNN index has {knn_count:,} vectors")
    assert knn_count == num_cells, (
        f"kNN count ({knn_count}) != label count ({num_cells})"
    )

    print(f"\nWriting Parquet to {OUTPUT_PATH} in batches of {BATCH_SIZE:,}...")
    writer = None

    for batch_start in tqdm(
        range(0, num_cells, BATCH_SIZE), desc="Processing", unit="batch"
    ):
        batch_end = min(batch_start + BATCH_SIZE, num_cells)
        batch_size = batch_end - batch_start

        # Read embeddings from kNN index
        indices = list(range(batch_start, batch_end))
        embeddings = knn.get_items(indices)  # (batch_size, 128) float32

        # Get labels and hierarchy for this batch
        batch_labels = labels[batch_start:batch_end]
        l1 = [hierarchy_map[lbl]["level1"] for lbl in batch_labels]
        l2 = [hierarchy_map[lbl]["level2"] for lbl in batch_labels]
        l3 = [hierarchy_map[lbl]["level3"] for lbl in batch_labels]
        l4 = [hierarchy_map[lbl]["level4"] for lbl in batch_labels]

        # Build PyArrow table
        flat_emb = embeddings.ravel().astype(np.float32)
        embedding_col = pa.FixedSizeListArray.from_arrays(
            pa.array(flat_emb, type=pa.float32()), list_size=EMBEDDING_DIM
        )

        table = pa.table(
            {
                "embedding": embedding_col,
                "level1_system": pa.array(l1, type=pa.utf8()).dictionary_encode(),
                "level2_lineage": pa.array(l2, type=pa.utf8()).dictionary_encode(),
                "level3_type": pa.array(l3, type=pa.utf8()).dictionary_encode(),
                "level4_subtype": pa.array(l4, type=pa.utf8()).dictionary_encode(),
            }
        )

        if writer is None:
            writer = pq.ParquetWriter(
                str(OUTPUT_PATH), table.schema, compression="snappy"
            )
        writer.write_table(table)

    if writer is not None:
        writer.close()

    # Validation
    print(f"\nValidating output...")
    pf = pq.ParquetFile(str(OUTPUT_PATH))
    metadata = pf.metadata
    print(f"  Rows: {metadata.num_rows:,}")
    print(f"  Columns: {metadata.num_columns}")
    print(f"  Row groups: {metadata.num_row_groups}")
    print(f"  File size: {OUTPUT_PATH.stat().st_size / (1024**3):.2f} GB")

    # Read back a small sample to verify
    sample = pq.read_table(str(OUTPUT_PATH), use_threads=False).slice(0, 5)
    print(f"\n  Sample (first 5 rows):")
    for i in range(5):
        emb = sample.column("embedding")[i].as_py()
        l1 = sample.column("level1_system")[i].as_py()
        l2 = sample.column("level2_lineage")[i].as_py()
        l3 = sample.column("level3_type")[i].as_py()
        l4 = sample.column("level4_subtype")[i].as_py()
        print(f"    [{i}] L1={l1}, L2={l2}, L3={l3}, L4={l4}, emb[:3]={emb[:3]}")

    # Print hierarchy distribution across all cells
    print(f"\n  Level 1 distribution:")
    full_table = pq.read_table(
        str(OUTPUT_PATH), columns=["level1_system"], use_threads=True
    )
    l1_col = full_table.column("level1_system").to_pandas()
    for val, count in l1_col.value_counts().items():
        print(f"    {val}: {count:,}")

    print("\nDone!")


if __name__ == "__main__":
    main()
