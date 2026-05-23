"""
Script 2: Generate Prompts for Cell-to-Sentence Embedding

Two feature selection strategies:
  ASPECT-2k:  Top-2000 genes for both CCLE and TCGA
  ASPECT-comb: Knowledge-based genes for CCLE, Top-2000 for TCGA

Usage:
    # ASPECT-2k mode
    python 2_gen_prompts.py --strategy ASPECT-2k --top_n_genes 2000

    # ASPECT-comb mode (gdsc knowledge)
    python 2_gen_prompts.py --strategy ASPECT-comb --dataset gdsc --top_n_genes 2000
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


def load_expression_from_feather(feather_path):
    """Load Celligner feather file and split into CCLE/TCGA DataFrames."""
    transcriptomics_df = pd.read_feather(feather_path).set_index('index')
    source_map = transcriptomics_df[['Source']]
    transcriptomics_df = transcriptomics_df.drop(columns=['Source'])
    ccle_expr = transcriptomics_df[source_map['Source'] == 'CCLE']
    tcga_expr = transcriptomics_df[source_map['Source'] == 'TCGA']
    return ccle_expr, tcga_expr


def generate_topn_prompts(expr_df, top_n, label_prefix, desc="samples"):
    """Generate Top-N gene expression prompts."""
    prompts_data = []
    for sample_id, expression_profile in tqdm(
        expr_df.iterrows(), total=len(expr_df), desc=f"Top-N {desc}"
    ):
        n = min(top_n, len(expression_profile))
        top_genes = expression_profile.nlargest(n)
        sentence = " ".join(top_genes.index.tolist())
        prompts_data.append({
            'SampleID': sample_id,
            'Prompt_Type': f'Top_{n}',
            'Prompt': sentence
        })
    return pd.DataFrame(prompts_data)


def generate_knowledge_prompts(ccle_expr, metadata_df, gene_getter, desc="CCLE Knowledge"):
    """Generate mechanism-based prompts for CCLE using drug-associated genes."""
    ccle_metadata = metadata_df[metadata_df['DepMapID'].isin(ccle_expr.index)]
    unique_drug_ids = ccle_metadata['DrugID'].unique()

    # Precompute drug-to-genes mapping
    drug_to_genes = {
        drug_id: gene_getter.get_genes(drug_id)
        for drug_id in unique_drug_ids
    }

    prompts_data = []
    for _, row in tqdm(ccle_metadata.iterrows(), total=len(ccle_metadata), desc=desc):
        sample_id = row['DepMapID']
        drug_id = row['DrugID']

        mechanism_genes = drug_to_genes.get(drug_id)
        if not mechanism_genes:
            continue

        valid_genes = [g for g in mechanism_genes if g in ccle_expr.columns]
        if not valid_genes:
            continue

        expression_profile = ccle_expr.loc[sample_id]
        mechanism_expression = expression_profile[valid_genes]
        sorted_genes = mechanism_expression.sort_values(ascending=False)
        sentence = " ".join(sorted_genes.index.tolist())

        prompts_data.append({
            'SampleID': sample_id,
            'DrugID': drug_id,
            'DrugName': row['Drug'],
            'Prompt_Type': 'Mechanism',
            'Prompt': sentence
        })

    return pd.DataFrame(prompts_data)


def main():
    parser = argparse.ArgumentParser(
        description="Generate prompts for C2S embedding with two strategies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        choices=['ASPECT-2k', 'ASPECT-comb'],
        help="Feature selection strategy: ASPECT-2k (Top2000 both) or ASPECT-comb (Knowledge CCLE + Top2000 TCGA)"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='gdsc',
        choices=['gdsc', 'prism'],
        help="Dataset for knowledge-based genes (ASPECT-comb only)"
    )
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
        '--output_path',
        type=str,
        default='./results',
        help="Path to save output files"
    )
    parser.add_argument(
        '--top_n_genes',
        type=int,
        default=2000,
        help="Number of top expressed genes for Top-N prompts"
    )
    parser.add_argument(
        '--filter_drug_names',
        type=str,
        default=None,
        help="JSON list of Drug Names to filter (ASPECT-comb only)"
    )

    args = parser.parse_args()

    start_time = time.time()
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Prompt Generation: {args.strategy} ---")

    # ========== Load expression data ==========
    print(f"Loading transcriptomics from: {args.celligner_path}")
    try:
        ccle_expr, tcga_expr = load_expression_from_feather(args.celligner_path)
    except Exception as e:
        print(f"Error: Could not load feather file: {e}")
        sys.exit(1)

    print(f"CCLE: {ccle_expr.shape[0]} samples x {ccle_expr.shape[1]} genes")
    print(f"TCGA: {tcga_expr.shape[0]} samples x {tcga_expr.shape[1]} genes")

    # ========== Generate CCLE prompts ==========
    if args.strategy == 'ASPECT-2k':
        # Top-2000 for each CCLE cell line
        print(f"\n--- CCLE: Top-{args.top_n_genes} genes ---")
        ccle_prompts_df = generate_topn_prompts(
            ccle_expr, args.top_n_genes, label_prefix=f"CCLE_Top{args.top_n_genes}", desc="CCLE cell lines"
        )
        ccle_output_path = output_dir / f'ccle_top{args.top_n_genes}_prompts.csv'

    else:  # ASPECT-comb
        # Knowledge-based genes for CCLE
        print(f"\n--- CCLE: Knowledge-based ({args.dataset}) ---")
        metadata_df = obtain_metadata(dataset=args.dataset, path=Path(args.data_path))

        if args.filter_drug_names:
            drugs_to_keep = json.loads(args.filter_drug_names)
            print(f"Filtering drugs: {drugs_to_keep}")
            metadata_df = metadata_df[metadata_df['Drug'].isin(drugs_to_keep)]
            print(f"After filtering: {len(metadata_df)} metadata entries")

        gene_getter = GeneGetter(
            dataset=args.dataset,
            data_path=args.data_path,
            available_genes=ccle_expr.columns
        )

        ccle_prompts_df = generate_knowledge_prompts(
            ccle_expr, metadata_df, gene_getter, desc=f"CCLE Knowledge ({args.dataset})"
        )
        ccle_output_path = output_dir / f'{args.dataset}_ccle_mechanism_prompts.csv'

    if len(ccle_prompts_df) > 0:
        ccle_prompts_df.to_csv(ccle_output_path, index=False)
        print(f"CCLE prompts saved: {ccle_output_path} ({len(ccle_prompts_df)} entries)")
    else:
        print("Warning: No CCLE prompts generated!")

    # ========== Generate TCGA prompts (always Top-N) ==========
    print(f"\n--- TCGA: Top-{args.top_n_genes} genes ---")
    tcga_prompts_df = generate_topn_prompts(
        tcga_expr, args.top_n_genes, label_prefix=f"TCGA_Top{args.top_n_genes}", desc="TCGA samples"
    )
    tcga_output_path = output_dir / f'tcga_top{args.top_n_genes}_prompts.csv'
    tcga_prompts_df.to_csv(tcga_output_path, index=False)
    print(f"TCGA prompts saved: {tcga_output_path} ({len(tcga_prompts_df)} entries)")

    end_time = time.time()
    print(f"\nDone. Total time: {end_time - start_time:.1f}s")


if __name__ == '__main__':
    main()
