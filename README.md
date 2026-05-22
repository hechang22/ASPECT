#ASPECT：A generalizable language model framework bridging cross-context transcriptomes for cancer therapy prediction

This repository contains the code for the manuscript "A generalizable language model framework bridging cross-context transcriptomes for cancer therapy prediction

## Overview
Predicting patient-specific drug response is vital for advancing precision oncology, as it enables the pre-emptive identification of effective therapeutic interventions while bypassing the practical constraints of clinical testing. However, although large-scale screens in cancer cell lines serve as the primary foundation for sensitivity modeling, the systematic divergence between cell line and tumor transcriptomes from real patients remains a major barrier to reliable cross-domain translation, limiting the fidelity of prediction transfer. Analogous to how words derive meaning from sentential context, genes acquire functional meaning within their transcriptomic context, allowing transcriptomes to be interpreted as a semantic language. We therefore propose ASPECT (Anchoring Semantic Phenotypes for Effective Cancer Treatment), a framework that leverages Large Language Models to align cross-context transcriptomes between patient tumors and cell lines. By characterizing semantic phenotypes including metabolic activity, proliferation, and malignancy, ASPECT matches patient samples to reference cell lines and reconstructs a drug benefit score quantifying therapeutic sensitivity. Specifically, we integrated transcriptomic profiles from 1,019 cell lines with drug sensitivity screening data across 286 drugs to interpret 14,197 multi-source patient samples, spanning a comprehensive spectrum of adult and pediatric malignancies across more than 80 cancer types. We demonstrated the model’s efficacy in chemotherapy sensitivity prediction, patient stratification and prognosis assessment, with an average AUC improvement of 0.35 over existing baselines. Furthermore, by analyzing 78 single-cell RNA (scRNA)-seq samples from 22 triple-negative breast cancer patients, we decoded extrinsic immune microenvironment dynamics. Our framework identifies patients with hot-but-suppressed microenvironments as ideal candidates for chemo-immunotherapy, providing a novel, interpretable strategy to guide precision oncology.


## Repository Structure

```
.
├── ASPECT/                     # Self-contained Python package
│   ├── __init__.py
│   ├── dataset_loaders.py      # Data loading and preprocessing
│   ├── gen_gene_list.py        # Mechanism-based gene selection
│   └── celligner.py            # Celligner integration utilities
├── scripts/                    # Analysis pipeline scripts
│   ├── 0_prepare_indications.py      # Clinical indications standardization
│   ├── 1_prepare_celligner.py        # Celligner alignment
│   ├── 2_gen_prompts.py              # Prompt generation
│   ├── 3_gen_embedding.py            # Embedding generation
│   ├── 4_predict_sensitivity.py      # Sensitivity prediction
│   ├── 5_validate_predictions.py     # Validation analysis
│   └── 6_analysis_pipeline.R         # Downstream R analysis
├── celligner2/                 # Celligner2 package (external dependency)
├── data/                       # Data directory (not in repo)
│   ├── metadata/               # Drug and sample metadata
│   ├── transcriptomics/        # Expression data
│   └── knowledge_data/         # Gene-drug associations
├── results/                    # Output directory
└── README.md                   # This file
```

## Dependencies

### Python Packages

Core dependencies (see `requirements.txt` for complete list):

- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- lightgbm >= 3.3.0
- torch >= 2.0.0
- transformers >= 4.30.0
- celligner2 (external, from https://github.com/broadinstitute/Celligner2)

### R Packages

- tidyverse, janitor, ggplot2, ggpubr
- TCGAbiolinks (TCGA subtype & clinical data)
- survival, survminer (survival analysis)
- forestmodel (Cox forest plots)
- mlr3, mlr3learners, mlr3viz, ranger (ML classification)
- psych (factor analysis)
- patchwork (figure composition)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd moa-c2s

# Install Python dependencies
pip install -r requirements.txt

# Install Celligner2 (external dependency)
pip install git+https://github.com/broadinstitute/Celligner2.git

# Install R packages
R -e "install.packages(c('tidyverse', 'janitor', 'ggplot2', 'ggpubr', 'TCGAbiolinks'))"
```

## Usage

### Step 1: Prepare Celligner Alignment

Align CCLE and TCGA transcriptomic data:

```bash
python scripts/1_prepare_celligner.py \
    --data_path ./data/transcriptomics \
    --output_path ./data/transcriptomics
```

### Step 2: Generate Prompts

Generate hybrid prompts (mechanism-based for CCLE, Top-N for TCGA):

```bash
python scripts/2_gen_prompts.py \
    --dataset gdsc \
    --data_path ./data \
    --celligner_path ./data/transcriptomics/celligner_CCLE_TCGA.feather \
    --output_path ./results \
    --top_n_genes 2000
```

For custom data:

```bash
python scripts/2_gen_prompts.py \
    --dataset others \
    --expression_csv ./data/custom_expression.csv \
    --output_path ./results \
    --top_n_genes 2000
```

### Step 3: Generate Embeddings

Generate embeddings using C2S-Scale model:

```bash
python scripts/3_gen_embedding.py \
    --model_path ./model/c2s-scale-gemma-2 \
    --prompts_file ./results/gdsc_ccle_mechanism_prompts.csv \
    --output_dir ./results/embeddings \
    --smart_batching
```

### Step 4: Predict Sensitivity

Predict drug sensitivity using k-NN (or GPR/LightGBM):

```bash
python scripts/4_predict_sensitivity.py \
    --ccle_prompts ./results/gdsc_ccle_mechanism_prompts.csv \
    --ccle_embeddings ./results/embeddings/ccle_embeddings.npy \
    --tcga_prompts ./results/tcga_top2000_prompts.csv \
    --tcga_embeddings ./results/embeddings/tcga_embeddings.npy \
    --dataset gdsc \
    --data_path ./data \
    --model_type knn \
    --k_neighbors 10 \
    --output_file ./results/predictions.csv
```

### Step 5: Validate Predictions

Validate predictions against clinical indications:

```bash
python scripts/5_validate_predictions.py \
    --predictions_csv ./results/predictions.csv \
    --phenotype ./data/metadata/clinical_TumorCompendium_v11_PolyA.tsv \
    --indications_file ./results/gdsc_clinical_indications.csv \
    --output_dir ./validation_results \
    --top_n 600
```

### Step 6: Downstream Analysis

Run the R analysis pipeline (15 sections covering model comparison, lineage analysis, tumor purity, molecular subtypes, survival analysis, CBI scoring, drug combination analysis, and more):

```r
source("scripts/6_analysis_pipeline.R")
# Edit file_list paths at top of script before running
```

The analysis script is organized into 15 sections:
1. **Model Comparison**: AUC & Recall bar charts across models
2. **Neighbor Lineage & Metastasis**: Chi-square test, lineage enrichment
3. **Cross-Cancer Lineage**: Bubble plot enrichment analysis
4. **Tumor Purity**: ESTIMATE purity vs predicted sensitivity
5. **Molecular Subtype Validation**: BRCA, OV, SKCM, PRAD subtypes
6. **CBI Subtype Validation**: Cytotoxic Burden Index across cancers
7. **Immune/Purity by BRCA Subtype**: ESTIMATE validation in breast cancer
8. **CCLE Primary vs Metastatic IC50**: Cell line validation
9. **ML Classification**: Random forest subtype prediction
10. **Survival Analysis**: KM curves + Cox regression (LGG, BRCA, SKCM)
11. **CBI Variants**: Mean/PCA/FA methods + survival evaluation
12. **BRCA Subtype-Specific Survival**: Stratified survival per subtype
13. **LGG Subtype & TOP2A Correlation**: Target engagement validation
14. **Drug Combination Co-sensitivity**: Regimen pair correlation
15. **Panel Figures**: Publication-ready combined figures

## Data Requirements

### Required Data Files

1. **Transcriptomic Data**:
   - CCLE: `OmicsExpressionProteinCodingGenesTPMLogp1.csv`
   - TCGA: `TumorCompendium_v11_PolyA_hugo_log2tpm_58581genes_2020-04-09.tsv`

2. **Metadata**:
   - `Model.csv`: CCLE cell line metadata
   - `GDSC2_fitted_dose_response_24Jul22.csv`: GDSC drug response data
   - `clinical_TumorCompendium_v11_PolyA.tsv`: TCGA clinical data

3. **Knowledge Data** (in `data/knowledge_data/`):
   - `{dataset}_LLM_drugID_to_genes.json`: LLM-predicted genes
   - `{dataset}_ligand_drugID_to_genes.json`: Ligand-associated genes
   - `{dataset}_target_drugID_to_genes.json`: Target-associated genes
   - `{dataset}_KEGG.json`: KEGG pathway genes
   - `{dataset}_important_genes.json`: Downstream pathway genes

4. **C2S-Scale Model**:
   - Download from [model repository]
   - Place in `./model/c2s-scale-gemma-2/`

## Key Features

### Mechanism-Aware Gene Selection

The `GeneGetter` class in `ASPECT/gen_gene_list.py` aggregates genes from multiple sources:
- LLM-predicted genes
- Ligand-associated genes
- Target-associated genes
- KEGG pathway genes
- Important/downstream pathway genes

### Multiple Regression Models

Script 4 supports three regression models:
- **k-NN**: K-nearest neighbors with distance weighting
- **GPR**: Gaussian Process Regression with uncertainty estimation
- **LightGBM**: Gradient boosting for non-linear relationships

### Validation Metrics

Script 5 computes:
- **Hypergeometric enrichment**: Tests if on-label samples are enriched in top predictions
- **AUC**: Area under ROC curve for binary classification
- **Recall@Top-N**: Fraction of on-label samples in top N predictions

## Notes on CellHit Dependency

This repository has been refactored to remove dependencies on the external `CellHit` package. All required functionality has been integrated into the `ASPECT` package:

- `obtain_metadata()`: Loads GDSC/PRISM metadata
- `GeneGetter`: Mechanism-based gene selection
- `IndexedArray`: Efficient array indexing
- `DatasetLoader`: Data splitting and scaling

The only remaining external dependency is `celligner2`, which is a published tool from the Broad Institute.

## Citation

If you use this code in your research, please cite:

```
[Manuscript citation to be added]
```

## License

[License information to be added]

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
