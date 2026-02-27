# Fractal Biological Embeddings

Let's see if applying [Devansh's](https://www.linkedin.com/in/devansh-devansh-516004168/) next level [Fractal Embeddings](https://www.artificialintelligencemadesimple.com/p/how-fractals-can-improve-how-ai-models) pumps our our biological semantic foo!

# SCimilarity Dataset

A hot bed of foundation model effort in the biology world starts with single-cell RNA-seq data - basically a 20k vector per cell where each row is the level of a gene as a proxy for a protein. [Genentech](https://www.gene.com/) (The OG Bio Tech Company) has published [SCimilarity](https://github.com/Genentech/scimilarity), a dataset of 23 million single cells and a single cell foundation model (scFM) that generates 128-dimensional embeddings per cell. The 7.9 million cell subset they used to train their model have ground-truth [Cell Ontology](https://obophenotype.github.io/cell-ontology/) labels spanning 203 unique cell types that we can mine to generate a hierarchy for our fractal embeddings. The full dataset is published as a TileDB with all 23M embeddings, and the labeled annotation subset is stored in an hnswlib kNN index alongside a reference labels file.

(Shameless plug - I've wrangled all these data plus the scFM model and an IVFPQ implementation into a 100% client side web app, [CytoVerse](https://github.com/braingeneers/cytoverse), but I digress...)

## Ingesting and Materializing a Hierarchy

The Cell Ontology (CL) is a Directed Acyclic Graph (DAG) with ~3,200 cell type terms connected by `is_a` relationships. Our 203 cell types sit at varying depths in this DAG. To create a strict 4-level tree suitable for hierarchical loss training, we:

1. **Map names to CL IDs** -- Parse `cl-basic.obo` with `obonet` and match each of the 203 cell type names to their CL identifiers via canonical names and synonyms. 201/203 match automatically; "native cell" and "animal cell" are obsolete terms requiring manual overrides.

2. **Define anchor nodes** at Level 1 (System) and Level 2 (Lineage) as fixed snap-points in the ontology:
   - **Level 1 (System):** Immune (`CL:0000988`), Epithelial (`CL:0000066`), Neural (`CL:0002319`), Endothelial (`CL:0000115`), Stromal (`CL:0002320`), Muscle (`CL:0000187`), with Stem/Progenitor (`CL:0000034`) as a fallback
   - **Level 2 (Lineage):** Lymphoid, Myeloid, Innate Lymphoid, Neuron, Glial, Fibroblast, Stromal, Adipocyte, Smooth/Cardiac/Skeletal Muscle

3. **Traverse the DAG** for each cell type using a two-pass anchor matching strategy:
   - For each CL ID, find all ontology ancestors via `nx.descendants()` (obonet edges are child-to-parent)
   - **Pass 1:** Find the nearest primary Level 1 anchor by shortest path length
   - **Pass 2:** Only if no primary anchor is reachable, fall back to Stem/Progenitor. This avoids the "stem cell trap" where many differentiated cells (monocytes, NK cells, etc.) have short paths to `stem cell` via `progenitor cell`
   - **Level 3 (Type):** The immediate parent on the shortest path from the leaf to its Level 2 anchor
   - **Level 4 (Subtype):** The original 203 cell type names

4. **Export** 7,913,892 embeddings from the hnswlib kNN annotation index with their 4-level hierarchy labels as a single Parquet file with dictionary-encoded categorical columns.

# Running

## Install

Install python dependencies and create a virtual env:

```
uv venv
source .venv/bin/activate
uv sync
```

Create a ./data/ folder and download and unpack the scimilarity [model and dataset](https://zenodo.org/records/10685499) (~30GB) into data/models/scimilarity/model_v1.1.

## Ingest

Convert the SCimilarity embeddings and flat labels and convert into a 4 level hierarchy.

```
uv run python ingest.py
```

**Output:** `data/scimilarity_embeddings.parquet` (3.80 GB)

| Column           | Type                            | Description                   |
| ---------------- | ------------------------------- | ----------------------------- |
| `embedding`      | `fixed_size_list<float32>[128]` | 128-dim SCimilarity embedding |
| `level1_system`  | `dictionary<string>`            | 8 categories                  |
| `level2_lineage` | `dictionary<string>`            | 17 categories                 |
| `level3_type`    | `dictionary<string>`            | ~120 categories               |
| `level4_subtype` | `dictionary<string>`            | 203 categories                |

**7,913,892 cells** across 203 cell types, mapped to 8 Level 1 systems:

| System          | Cells     | %     |
| --------------- | --------- | ----- |
| Immune          | 4,308,236 | 54.4% |
| Epithelial      | 956,129   | 12.1% |
| Neural          | 635,314   | 8.0%  |
| Stromal         | 544,481   | 6.9%  |
| Other           | 466,874   | 5.9%  |
| Muscle          | 396,372   | 5.0%  |
| Endothelial     | 385,952   | 4.9%  |
| Stem/Progenitor | 220,534   | 2.8%  |

## Train

```
uv run python train.py --max-samples 100000 --epochs 20

```

**Output:**

```
Device: mps
Loading data from data/scimilarity_embeddings.parquet...
  Total rows: 7,913,892
  L0 classes (system): 8
  L1 classes (subtype): 203
  Stratified subsampling to 100,000 rows...
  Subsampled: 100,000 rows
  Train: 85,000, Val: 15,000
FractalHeadV5: 259,795 parameters
  scales=2, scale_dim=64, output_dim=128
  L0 classes=8, L1 classes=203
Epoch 1/20: 100%|██| 332/332 [19:51<00:00,  3.59s/it, loss=8.5752]
  Epoch 1: loss=10.0869, L0=0.9500, L1=0.6840
...
Epoch 20/20: 100%|█| 332/332 [19:59<00:00,  3.61s/it, loss=5.6032]
  Epoch 20: loss=5.4599, L0=0.9615, L1=0.7610
  Early stopping at epoch 20

Saved model to data/fractal_adapter.pt
Best score: 1.7230
```

## Evaluate

```
uv run python eval.py
```

**Output:**

```
Loading model from data/fractal_adapter.pt...
Loading data from data/scimilarity_embeddings.parquet...
  Evaluating on 5,000 samples

--- Original Embeddings (128d) ---
  kNN L0 accuracy (system):  0.9404
  kNN L1 accuracy (subtype): 0.7574
  Silhouette (L0, cosine):   0.3054

--- Fractal Embeddings ---
  Full (128d) kNN L0 accuracy:  0.9464
  Full (128d) kNN L1 accuracy:  0.7910
  Full (128d) Silhouette (L0):  0.2445
  Prefix (64d) kNN L0 accuracy: 0.9460
  Prefix (64d) kNN L1 accuracy: 0.7912
  Prefix (64d) Silhouette (L0): 0.5495

--- Steerability ---
  S = (L0@64d - L0@128d) + (L1@128d - L1@64d)
  S = (0.9460 - 0.9464) + (0.7910 - 0.7912)
  S = -0.0006
  Non-positive steerability: prefix not yet specializing for coarse labels.

============================================================
Embedding            Dims   L0 Acc     L1 Acc     Sil (L0)
------------------------------------------------------------
Original             128    0.9404     0.7574     0.3054
Fractal (full)       128    0.9464     0.7910     0.2445
Fractal (prefix)     64     0.9460     0.7912     0.5495
============================================================
```
