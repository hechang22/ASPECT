"""
Script 5: Validate Predictions using Clinical Indications

This script validates drug sensitivity predictions using:
- Clinical indication data (FDA-approved indications)
- Hypergeometric enrichment test
- AUC analysis

Usage:
    python 5_validate_predictions.py --predictions_csv predictions.csv --output_dir ./validation
"""

import pandas as pd
import numpy as np
import argparse
from scipy.stats import hypergeom
from sklearn.metrics import roc_auc_score
from pathlib import Path
from tqdm import tqdm
import ast


def load_clinical_indications(indications_file_path):
    """
    Load standardized clinical indications from CSV.
    
    Args:
        indications_file_path: Path to indications CSV
        
    Returns:
        Dictionary mapping drug names to lists of indications
    """
    print(f"--- Loading clinical indications from: {indications_file_path} ---")
    try:
        df = pd.read_csv(indications_file_path)
        # Column from 0_prepare_indications.py output: Standardized_Indications, GDSC_DrugName
        col_indications = 'Standardized_Indications'
        col_drugname = 'GDSC_DrugName'
        
        if df[col_indications].dtype == 'object':
             df[col_indications] = df[col_indications].apply(ast.literal_eval)
        
        indications_dict = pd.Series(
            df[col_indications].values,
            index=df[col_drugname]
        ).to_dict()
        print(f"Successfully created dictionary for {len(indications_dict)} drugs.")
        return indications_dict
    except Exception as e:
        print(f"Error: Could not load indications file. {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Validate predictions using clinical indications and enrichment analysis."
    )
    
    parser.add_argument(
        '--predictions_csv', 
        type=str,
        default="./results/predictions.csv",
        help="Path to predictions CSV file"
    )
    parser.add_argument(
        '--phenotype', 
        type=str,
        default="./data/metadata/clinical_TumorCompendium_v11_PolyA.tsv",
        help="Path to TCGA phenotype file"
    )
    parser.add_argument(
        '--indications_file', 
        type=str,
        default="./results/gdsc_clinical_indications.csv",
        help="Path to standardized indications CSV"
    )
    parser.add_argument(
        '--output_dir', 
        type=str,
        default='./validation_results',
        help="Directory to save validation outputs"
    )
    parser.add_argument(
        '--top_n', 
        type=int, 
        default=600,
        help="Top N samples for enrichment analysis"
    )
   
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("--- 1. Loading All Data ---")
    try:
        predictions_df = pd.read_csv(args.predictions_csv)
        print(f"Loaded {len(predictions_df)} total predictions.")
        
        phenotype_df = pd.read_csv(args.phenotype, sep='\t')
        id_col_in_pheno = 'th_dataset_id'
        patient_to_disease_map = phenotype_df[[id_col_in_pheno, 'disease']].drop_duplicates()
        
        clinical_indications = load_clinical_indications(args.indications_file)
        if not clinical_indications:
            print("Clinical indications dictionary is empty. Exiting.")
            return

    except Exception as e:
        print(f"Error during data loading: {e}")
        return

    # Prepare analysis DataFrame
    print("\n--- 2. Preparing Analysis DataFrame ---")
    
    analysis_df = pd.merge(
        predictions_df,
        patient_to_disease_map,
        left_on='SampleID',
        right_on=id_col_in_pheno,
        how='left'
    ).dropna(subset=['disease'])
    
    print(f"Successfully annotated {len(analysis_df)} predictions with disease types.")
    
    # Validate by drug
    print(f"\n--- 3. Starting Validation for {analysis_df['DrugName'].nunique()} drugs ---")
    
    all_results = []
    ground_truth_list = []
    
    for drug_name, drug_df in tqdm(analysis_df.groupby('DrugName'), desc="Validating Drugs"):
        
        if drug_name not in clinical_indications or not clinical_indications[drug_name]:
            continue
        
        on_label_diseases = clinical_indications[drug_name]

        # Calculate ground truth labels
        drug_df['y_true'] = drug_df['disease'].isin(on_label_diseases).astype(int)
        
        # Collect ground truth
        current_truth = drug_df[['SampleID', 'y_true']].copy()
        current_truth['DrugName'] = drug_name
        ground_truth_list.append(current_truth)
        
        # Top-N Enrichment Analysis
        # NOTE: ascending=False picks samples with HIGHEST predicted IC50 (most sensitive)
        drug_df_sorted = drug_df.sort_values('Predicted_IC50', ascending=False) 
        top_n_df = drug_df_sorted.head(args.top_n)

        M = len(drug_df)
        n = drug_df['y_true'].sum()
        N = len(top_n_df)
        k = top_n_df['disease'].isin(on_label_diseases).sum()
        
        if n < 3:
            continue
        if N == 0:
            continue

        p_val_hypergeom = hypergeom.sf(k - 1, M, n, N)
        recall_at_top_n = k / n
        
        # AUC Analysis
        drug_df['y_score'] = -drug_df['Predicted_IC50']
        
        auc_score = roc_auc_score(
            drug_df['y_true'], 
            drug_df['y_score']
        ) if drug_df['y_true'].nunique() > 1 else np.nan

        # Compile results
        result = {
            'DrugName': drug_name,
            'Total_TCGA_Samples': M,
            'Total_On_Label_Samples': n,
            'Recall_at_Top_N': recall_at_top_n,
            'Hypergeometric_P_Value': p_val_hypergeom,
            'AUC': auc_score,
            'Is_Enriched': p_val_hypergeom < 0.05,
            'Is_Predictive': auc_score > 0.5 if not np.isnan(auc_score) else False
        }
        all_results.append(result)

    # Save validation summary
    if not all_results:
        print("\nNo validation could be performed.")
    else:
        final_results_df = pd.DataFrame(all_results).sort_values('Hypergeometric_P_Value')
        
        print("\n--- Overall Validation Summary ---")
        print(final_results_df.head().to_string())
        
        enrichment_success_rate = final_results_df['Is_Enriched'].mean()
        auc_success_rate = final_results_df.dropna(subset=['Is_Predictive'])['Is_Predictive'].mean()
        
        print(f"\nSuccess Rate (Enrichment, p < 0.05): {enrichment_success_rate:.2%}")
        print(f"Success Rate (AUC > 0.5): {auc_success_rate:.2%}")
        
        output_csv_path = output_dir / "validation_summary.csv"
        final_results_df.to_csv(output_csv_path, index=False)
        print(f"\nSummary results saved to: {output_csv_path}")
        
        # Save ground truth sensitivity matrix
        if ground_truth_list:
            print("\nGenerating Ground Truth Matrix...")
            all_truth_df = pd.concat(ground_truth_list, ignore_index=True)
            
            truth_matrix = all_truth_df.pivot(
                index='DrugName', 
                columns='SampleID', 
                values='y_true'
            )
            
            truth_matrix = truth_matrix.fillna(0).astype(int)
            
            matrix_output_path = output_dir / "ground_truth_sensitivity_matrix.csv"
            truth_matrix.to_csv(matrix_output_path)
            print(f"Ground truth matrix saved to: {matrix_output_path}")

    print(f"\nValidation complete.")


if __name__ == '__main__':
    main()
