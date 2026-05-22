# ASPECT Repository Structure

## Summary of Changes

This repository has been organized for manuscript submission. The key change is the **removal of CellHit dependency** - all required functionality has been integrated into the self-contained `ASPECT` package.

## Directory Structure

```
workspace/
├── ASPECT/                     # Self-contained Python package (NO CellHit dependency)
│   ├── __init__.py             # Package initialization with exports
│   ├── dataset_loaders.py      # Data loading: obtain_metadata, DatasetLoader, IndexedArray
│   ├── gen_gene_list.py        # Mechanism-based gene selection: GeneGetter
│   └── celligner.py            # Celligner utilities
│
├── scripts/                    # Analysis pipeline (numbered in execution order)
│   ├── 0_prepare_indications.py      # Standardize clinical indications
│   ├── 1_prepare_celligner.py        # Celligner alignment
│   ├── 2_gen_prompts.py              # Generate text prompts
│   ├── 3_gen_embedding.py            # Generate C2S embeddings
│   ├── 4_predict_sensitivity.py      # Predict drug sensitivity
│   ├── 5_validate_predictions.py     # Validate with clinical indications
│   └── 6_analysis_pipeline.R         # R downstream analysis
│
├── celligner2/                 # External dependency (Broad Institute)
│   └── ...                     # Published tool, kept as-is
│
├── README.md                   # Main documentation
├── requirements.txt            # Python dependencies
└── REPOSITORY_STRUCTURE.md     # This file

## Original files (kept for reference, not used in pipeline):
├── 1_prepare_celligner_script.py
├── 1_prepare_standardize_indications.py
├── 2_gen_prompts_v2.py
├── 3_gen_embedding.py
├── 4_correlation_study_v2.py
├── 5_validate.py
└── 6_analysis_pipeline.R (original)
```

## Key Changes

### 1. Removed CellHit Dependency

**Before:** Scripts imported from `CellHit.data`
```python
from CellHit.data import obtain_metadata  # OLD - external dependency
```

**After:** Scripts import from `ASPECT`
```python
from ASPECT.dataset_loaders import obtain_metadata  # NEW - self-contained
```

### 2. Integrated Functions into ASPECT

The following functions from CellHit have been integrated:

| Function | Original Location | New Location |
|----------|------------------|--------------|
| `obtain_metadata()` | `CellHit.data.metadata_processing` | `ASPECT.dataset_loaders` |
| `obtain_gdsc()` | `CellHit.data.metadata_processing` | `ASPECT.dataset_loaders` |
| `obtain_prism_lfc()` | `CellHit.data.metadata_processing` | `ASPECT.dataset_loaders` |
| `GeneGetter` | `CellHit.data.metadata_processing` | `ASPECT.gen_gene_list` |
| `IndexedArray` | `CellHit.data.indexed_array` | `ASPECT.dataset_loaders` |
| `DatasetLoader` | `CellHit.data.dataset_loaders` | `ASPECT.dataset_loaders` |

### 3. Script Improvements

All scripts have been updated with:
- **Docstrings**: Comprehensive module and function documentation
- **Type hints**: Where appropriate for clarity
- **Argument parsing**: Better CLI with help messages
- **Error handling**: Try-except blocks for file operations
- **Path handling**: Using `pathlib.Path` for cross-platform compatibility
- **Comments**: Chinese comments translated to English

### 4. R Script Modularization

The original `6_analysis_pipeline.R` was a monolithic script. It has been refactored into:

- `compare_models()`: Model comparison (AUC and Recall)
- `analyze_neighbor_lineage()`: Neighbor lineage enrichment
- `analyze_cross_cancer_lineage()`: Cross-cancer lineage analysis
- `analyze_tumor_purity()`: Tumor purity correlation
- `analyze_molecular_subtypes()`: Molecular subtype analysis
- `run_complete_analysis()`: Run all analyses for a drug

## Execution Order

Scripts are numbered to indicate execution order:

1. **0_prepare_indications.py**: Prepare clinical indication mapping
2. **1_prepare_celligner.py**: Align transcriptomic data
3. **2_gen_prompts.py**: Generate text prompts from expression
4. **3_gen_embedding.py**: Generate embeddings using C2S-Scale
5. **4_predict_sensitivity.py**: Predict drug sensitivity
6. **5_validate_predictions.py**: Validate predictions
7. **6_analysis_pipeline.R**: Downstream analysis and visualization

## Dependencies

### Required (Python)
- pandas, numpy, scipy
- scikit-learn, lightgbm
- torch, transformers, bitsandbytes
- celligner2 (external)

### Required (R)
- tidyverse, janitor, ggplot2, ggpubr
- TCGAbiolinks (for subtype analysis)

## Usage Example

```bash
# Step 0: Prepare indications
python scripts/0_prepare_indications.py \
    --nci_input ./data/metadata/nci_compiled_dataset.csv \
    --gdsc_mapping ./data/metadata/gdsc_pubchem_mappings.csv \
    --output_csv ./results/gdsc_clinical_indications.csv

# Step 1: Celligner alignment
python scripts/1_prepare_celligner.py \
    --data_path ./data/transcriptomics \
    --output_path ./data/transcriptomics

# Step 2: Generate prompts
python scripts/2_gen_prompts.py \
    --dataset gdsc \
    --data_path ./data \
    --output_path ./results \
    --top_n_genes 2000

# Step 3: Generate embeddings
python scripts/3_gen_embedding.py \
    --model_path ./model/c2s-scale-gemma-2 \
    --prompts_file ./results/gdsc_ccle_mechanism_prompts.csv \
    --output_dir ./results/embeddings \
    --smart_batching

# Step 4: Predict sensitivity
python scripts/4_predict_sensitivity.py \
    --ccle_prompts ./results/gdsc_ccle_mechanism_prompts.csv \
    --ccle_embeddings ./results/embeddings/ccle_embeddings.npy \
    --tcga_prompts ./results/tcga_top2000_prompts.csv \
    --tcga_embeddings ./results/embeddings/tcga_embeddings.npy \
    --output_file ./results/predictions.csv

# Step 5: Validate
python scripts/5_validate_predictions.py \
    --predictions_csv ./results/predictions.csv \
    --output_dir ./validation_results

# Step 6: R analysis
R -e "source('scripts/6_analysis_pipeline.R'); run_complete_analysis(list(drug_name='Vinblastine'))"
```

## Notes

- The `CellHit/` directory is kept for reference but is **not used** by the pipeline
- All scripts now use relative paths by default (configurable via arguments)
- The `ASPECT` package is fully self-contained and can be imported independently
