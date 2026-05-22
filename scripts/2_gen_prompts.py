"""
Script 2: Generate Prompts for Cell-to-Sentence Embedding

This script generates hybrid prompts combining:
- Mechanism-based prompts for CCLE samples (using drug-associated genes)
- Top-N gene prompts for target samples (TCGA or custom data)

Usage:
    python 2_gen_prompts.py --dataset gdsc --top_n_genes 2000
    python 2_gen_prompts.py --dataset others --expression_csv custom_data.csv
"""

import pandas as pd
from pathlib import Path
import argparse
import time
import json
import sys
from tqdm import tqdm

# Import from ASPECT (self-contained, no CellHit dependency)
from ASPECT.dataset_loaders import obtain_metadata
from ASPECT.gen_gene_list import GeneGetter


def load_custom_csv(csv_path):
    """
    Load custom CSV file with expression data.
    
    Expected input format: Rows = Genes, Columns = Samples
    Output format: Rows = Samples, Columns = Genes (standard pandas format)
    """
    print(f"Loading custom expression CSV from: {csv_path}")
    try:
        df = pd.read_csv(csv_path, index_col=0)
        df = df.T  # Transpose to (Samples x Genes)
        print(f"Loaded custom data with {df.shape[0]} samples and {df.shape[1]} genes.")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate HYBRID prompts with filtering: Mechanism-based for CCLE, Top-N for Target Samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset selection
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True, 
        choices=['gdsc', 'prism', 'others'],
        help="Dataset to use: 'gdsc' or 'prism' for mechanism-based prompts, 'others' for Top-N only"
    )
    
    # Paths
    parser.add_argument(
        '--data_path', 
        type=str, 
        default='./data',
        help="Path to data directory"
    )
    parser.add_argument(
        '--celligner_path', 
        type=str, 
        default='./data/transcriptomics/celligner_CCLE_TCGA.feather',
        help="Path to Celligner output feather file"
    )
    parser.add_argument(
        '--expression_csv', 
        type=str, 
        default=None,
        help="Path to custom expression CSV (Rows=Genes, Cols=Samples). Replaces TCGA data if provided."
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='./results',
        help="Path to save output files"
    )
    
    # Parameters
    parser.add_argument(
        '--top_n_genes', 
        type=int, 
        default=2000, 
        help="Number of top expressed genes for target samples"
    )
    
    # Filtering options
    parser.add_argument(
        '--filter_projects', 
        type=str, 
        default=None, 
        help="JSON list of TCGA project_ids to keep (only for standard TCGA data)"
    )
    parser.add_argument(
        '--filter_drug_names', 
        type=str, 
        default=None, 
        help="JSON list of Drug Names to filter (only for GDSC/PRISM)"
    )
    
    args = parser.parse_args()

    start_time = time.time()
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Starting Hybrid Prompt Generation (Dataset: {args.dataset}) ---")

    # --- 1. Data Loading ---
    ccle_expr = pd.DataFrame()
    tcga_expr = pd.DataFrame()
    
    if args.dataset == 'others':
        # Mode A: Others - Custom CSV only
        if not args.expression_csv:
            print("Error: For dataset 'others', you must provide --expression_csv.")
            sys.exit(1)
        
        tcga_expr = load_custom_csv(args.expression_csv)
        print("Dataset is 'others': Skipping CCLE/Mechanism loading.")
        
    else:
        # Mode B: GDSC / PRISM
        print(f"Loading transcriptomics matrix from: {args.celligner_path}")
        try:
            transcriptomics_df = pd.read_feather(args.celligner_path).set_index('index')
            source_map = transcriptomics_df[['Source']]
            transcriptomics_df = transcriptomics_df.drop(columns=['Source'])
            
            # Extract CCLE for mechanism prompts
            ccle_expr = transcriptomics_df[source_map['Source'] == 'CCLE']
            tcga_expr_default = transcriptomics_df[source_map['Source'] == 'TCGA']
        except Exception as e:
            print(f"Warning: Could not load Feather file: {e}")
            if not args.expression_csv:
                print("Error: No Feather file and no CSV provided. Cannot proceed.")
                sys.exit(1)

        # Determine target data source
        if args.expression_csv:
            print("Custom CSV provided. Using CSV data as Target samples.")
            tcga_expr = load_custom_csv(args.expression_csv)
        else:
            print("Using standard TCGA data from Feather file.")
            tcga_expr = tcga_expr_default

        print(f"Loaded {len(ccle_expr)} CCLE samples and {len(tcga_expr)} Target samples.")

    # --- 2. Load Metadata and Prepare Gene Mapping (GDSC/PRISM only) ---
    metadata_df = pd.DataFrame()
    drug_to_genes_ccle = {}

    if args.dataset != 'others':
        print("Loading metadata...")
        metadata_df = obtain_metadata(dataset=args.dataset, path=Path(args.data_path))
        
        # Filter drugs if specified
        if args.filter_drug_names:
            try:
                drugs_to_keep = json.loads(args.filter_drug_names)
                print(f"Filtering for drugs: {drugs_to_keep}")
                metadata_df = metadata_df[metadata_df['Drug'].isin(drugs_to_keep)]
                print(f"After filtering, {len(metadata_df)} metadata pairs remain.")
            except Exception as e:
                print(f"Error parsing drug filter: {e}")

        # Prepare GeneGetter
        gene_getter = GeneGetter(
            dataset=args.dataset,
            data_path=args.data_path,
            available_genes=ccle_expr.columns
        )
        
        # Preprocess CCLE drug-gene mapping
        ccle_metadata = metadata_df[metadata_df['DepMapID'].isin(ccle_expr.index)]
        unique_drug_ids_ccle = ccle_metadata['DrugID'].unique()
        drug_to_genes_ccle = {
            drug_id: gene_getter.get_genes(drug_id) 
            for drug_id in unique_drug_ids_ccle
        }
    
    # Filter TCGA samples by project if specified
    if args.filter_projects and args.dataset != 'others' and not args.expression_csv:
        try:
            projects_to_keep = json.loads(args.filter_projects)
            print(f"Filtering TCGA samples for projects: {projects_to_keep}")
            
            tcga_meta_path = Path(args.data_path) / 'metadata' / 'tcga_clinical.tsv'
            if tcga_meta_path.exists():
                tcga_meta = pd.read_csv(
                    tcga_meta_path, 
                    sep='\t', 
                    usecols=['case_submitter_id', 'project_id']
                )
                id_to_project_map = tcga_meta.set_index('case_submitter_id')['project_id']
                
                tcga_expr_temp = tcga_expr.copy()
                tcga_expr_temp['project_id'] = tcga_expr_temp.index.map(id_to_project_map)
                tcga_expr = tcga_expr_temp[tcga_expr_temp['project_id'].isin(projects_to_keep)]
                tcga_expr = tcga_expr.drop(columns=['project_id'])
                print(f"After filtering, {len(tcga_expr)} TCGA samples remain.")
            else:
                print(f"Warning: Metadata file not found at {tcga_meta_path}")
        except Exception as e:
            print(f"Warning: Could not filter TCGA by project. Error: {e}")

    # --- 3. Generate Prompts ---
    
    # 3.1 CCLE Mechanism-based Prompts (GDSC/PRISM only)
    if args.dataset != 'others' and not ccle_expr.empty:
        print("\n--- Generating Mechanism-based Prompts for CCLE ---")
        
        ccle_prompts_data = []
        ccle_metadata = metadata_df[metadata_df['DepMapID'].isin(ccle_expr.index)]

        for _, row in tqdm(ccle_metadata.iterrows(), total=len(ccle_metadata), desc="CCLE Pairs"):
            sample_id = row['DepMapID']
            drug_id = row['DrugID']
            
            mechanism_genes = drug_to_genes_ccle.get(drug_id)
            if not mechanism_genes:
                continue
            
            valid_genes = [g for g in mechanism_genes if g in ccle_expr.columns]
            if not valid_genes:
                continue

            expression_profile = ccle_expr.loc[sample_id]
            mechanism_expression = expression_profile[valid_genes]
            sorted_genes = mechanism_expression.sort_values(ascending=False)
            sentence = " ".join(sorted_genes.index.tolist())
            
            ccle_prompts_data.append({
                'SampleID': sample_id,
                'DrugID': drug_id,
                'DrugName': row['Drug'],
                'Prompt_Type': 'Mechanism',
                'Prompt': sentence
            })

        if ccle_prompts_data:
            ccle_prompts_df = pd.DataFrame(ccle_prompts_data)
            print(f"\nGenerated {len(ccle_prompts_df)} mechanism-based prompts for CCLE.")
            ccle_output_path = output_dir / f'{args.dataset}_ccle_mechanism_prompts.csv'
            ccle_prompts_df.to_csv(ccle_output_path, index=False)
            print(f"CCLE prompts saved to: {ccle_output_path}")

    # 3.2 Target/TCGA Top-N Prompts
    if not tcga_expr.empty:
        dataset_label = "CustomCSV" if args.expression_csv or args.dataset == 'others' else "TCGA"
        print(f"\n--- Generating Top-{args.top_n_genes} Gene Prompts for {dataset_label} ---")
        
        tcga_prompts_data = []
        
        for sample_id, expression_profile in tqdm(
            tcga_expr.iterrows(), 
            total=len(tcga_expr), 
            desc=f"Processing {dataset_label} Samples"
        ):
            n = min(args.top_n_genes, len(expression_profile))
            top_n_genes = expression_profile.nlargest(n)
            sentence = " ".join(top_n_genes.index.tolist())
            
            tcga_prompts_data.append({
                'SampleID': sample_id,
                'DrugID': 'N/A',
                'DrugName': 'N/A',
                'Prompt_Type': f'Top_{n}',
                'Prompt': sentence
            })

        if tcga_prompts_data:
            tcga_prompts_df = pd.DataFrame(tcga_prompts_data)
            print(f"Generated {len(tcga_prompts_df)} top-N-based prompts for {dataset_label}.")
            
            if args.dataset == 'others':
                target_output_filename = f'custom_top{args.top_n_genes}_prompts.csv'
            elif args.expression_csv:
                target_output_filename = f'{args.dataset}_custom_target_top{args.top_n_genes}_prompts.csv'
            else:
                target_output_filename = f'tcga_top{args.top_n_genes}_prompts.csv'

            tcga_output_path = output_dir / target_output_filename
            tcga_prompts_df.to_csv(tcga_output_path, index=False)
            print(f"{dataset_label} prompts saved to: {tcga_output_path}")
    
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds.")


if __name__ == '__main__':
    main()
