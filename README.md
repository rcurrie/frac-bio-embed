# Fractal Biological Embeddings

Let's see if applying [Devansh's](https://www.linkedin.com/in/devansh-devansh-516004168/) next level [Fractal Embeddings](https://www.artificialintelligencemadesimple.com/p/how-fractals-can-improve-how-ai-models) pumps our our biological semantic foo!

# SCimilarity Dataset

A hot bed of foundation model effort in the biology world starts with single-cell RNA-seq data - basically a 20k vector per cell where each row is the level of a gene as a proxy for a protein. [Genentech](https://www.gene.com/) (The OG Bio Tech Company) has published [SCimilarity](https://github.com/Genentech/scimilarity), a dataset of 23 million single cells and a single cell foundation model (scFM) that generates 128-dimensional embeddings per cell. The 7.9 million cell subset they used to train their model have ground-truth [Cell Ontology](https://obophenotype.github.io/cell-ontology/) labels spanning 203 unique cell types that we can mine to generate a hierarchy for our fractal embeddings. The full dataset is published as a TileDB with all 23M embeddings, and the labeled annotation subset is stored in an hnswlib kNN index alongside a reference labels file.

(Shameless plug - I've wrangled all these data plus the scFM model and an IVFPQ implementation into a 100% client side web app, [CytoVerse](https://github.com/braingeneers/cytoverse), but I digress...)

## Materializing The Hierarchy

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

## Results

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
