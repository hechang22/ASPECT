# ASPECT Repository Structure

## Summary

This repository has been organized for manuscript submission. The key change is the **removal of CellHit dependency** — all required functionality has been integrated into the `utils` package. The only remaining external dependency is `celligner` (script 1) / `celligner2` (bundled).

## Directory Structure

```
workspace/
├── utils/                       # Utility package
│   ├── __init__.py              # Package initialization with exports
│   ├── dataset_loaders.py       # obtain_metadata, DatasetLoader, IndexedArray
│   ├── gen_gene_list.py         # GeneGetter (mechanism-based gene selection)
│   └── celligner.py             # Simplified Celligner utilities
│
├── scripts/                     # Analysis pipeline (numbered by execution order)
│   ├── 0_prepare_indications.py       # Standardize clinical indications
│   ├── 1_prepare_celligner.py         # Celligner CCLE-TCGA alignment
│   ├── 2_gen_prompts.py               # Prompt generation (ASPECT-2k / ASPECT-comb)
│   ├── 3_gen_embedding.py             # C2S-Scale embedding generation
│   ├── 4_predict_sensitivity.py       # Drug sensitivity prediction (k-NN/GPR/LGBM)
│   └── 5_validate_predictions.py      # Validation with clinical indications
│
├── examples/                    # Usage examples
│   ├── 6_analysis.R              # Downstream data processing (saves workspace)
│   └── 6_figures.qmd             # Downstream figures (Quarto notebook)
│
├── original_scripts/            # Original scripts (reference only)
├── celligner2/                  # Celligner2 (external, Broad Institute)
├── framework.png                # ASPECT framework diagram
├── .gitignore
├── README.md
├── requirements.txt
└── REPOSITORY_STRUCTURE.md
```

## Key Changes

### 1. Removed CellHit Dependency

**Before:** Scripts imported from `CellHit.data`
```python
from CellHit.data import obtain_metadata  # OLD
```

**After:** Scripts import from `utils`
```python
from utils.dataset_loaders import obtain_metadata  # NEW
```

### 2. Integrated Functions

| Function | Original Location | New Location |
|----------|------------------|--------------|
| `obtain_metadata()` | `CellHit.data.metadata_processing` | `utils.dataset_loaders` |
| `obtain_gdsc()` | `CellHit.data.metadata_processing` | `utils.dataset_loaders` |
| `obtain_prism_lfc()` | `CellHit.data.metadata_processing` | `utils.dataset_loaders` |
| `GeneGetter` | `CellHit.data.metadata_processing` | `utils.gen_gene_list` |
| `IndexedArray` | `CellHit.data.indexed_array` | `utils.dataset_loaders` |
| `DatasetLoader` | `CellHit.data.dataset_loaders` | `utils.dataset_loaders` |

### 3. Two Prompt Strategies

| Strategy | CCLE | TCGA |
|----------|------|------|
| `ASPECT-2k` | Top-2000 genes per cell line | Top-2000 per sample |
| `ASPECT-comb` | Knowledge-based (LLM + ligand + target + KEGG) | Top-2000 per sample |

### 4. Dual Prediction Modes

`4_predict_sensitivity.py` supports both CCLE prompt formats via `--ccle_strategy`:
- `topn` — CCLE prompts lack DrugID; metadata matched by cell line
- `knowledge` — CCLE prompts have DrugID/DrugName; direct merge

## Execution Order

```
0_prepare_indications.py       → gdsc_clinical_indications.csv
1_prepare_celligner.py         → celligner_CCLE_TCGA.feather
2_gen_prompts.py               → ccle_top2000_prompts.csv / tcga_top2000_prompts.csv
3_gen_embedding.py             → ccle_embeddings.npy / tcga_embeddings.npy
4_predict_sensitivity.py       → predictions.csv
5_validate_predictions.py      → validation_summary.csv
6_analysis_pipeline.R          → figures & tables
```

## Dependencies

### Python
- pandas, numpy, scipy, scikit-learn, lightgbm
- torch, transformers, bitsandbytes (for step 3)
- celligner (for step 1)

### R
- tidyverse, janitor, ggplot2, ggpubr, TCGAbiolinks
- survival, survminer, forestmodel, mlr3, ranger, psych, patchwork

## Notes

- The `CellHit/` directory is kept for reference but is **not used** by the pipeline
- All scripts use relative paths by default (configurable via CLI arguments)
- The `utils` package is fully self-contained and independently importable
