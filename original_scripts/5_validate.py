# run_validation_single_file.py

import pandas as pd
import numpy as np
import argparse
from scipy.stats import hypergeom
from sklearn.metrics import roc_auc_score
from pathlib import Path
from tqdm import tqdm
import ast

def load_clinical_indications_from_file(indications_file_path):
    """
    Reads the standardized indications CSV and converts it into a dictionary.
    """
    print(f"--- Loading clinical indications from: {indications_file_path} ---")
    try:
        df = pd.read_csv(indications_file_path)
        if df['DrugIndicationsStandardized'].dtype == 'object':
             df['DrugIndicationsStandardized'] = df['DrugIndicationsStandardized'].apply(ast.literal_eval)
        # 使用 DrugName 作为键
        indications_dict = pd.Series(
            df['DrugIndicationsStandardized'].values,
            index=df['DrugName']
        ).to_dict()
        print(f"Successfully created a dictionary for {len(indications_dict)} drugs.")
        return indications_dict
    except Exception as e:
        print(f"Error: Could not load or parse the indications file. {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Validate a single prediction file using Top-N enrichment and AUC analysis.")
    
    parser.add_argument('--predictions_csv', default="/Users/hechang/Documents/chenlab/CellHit_onlyCPU/scripts/feat_C2s/predictions_comb.csv", help="Path to the SINGLE CSV file containing all predictions.")
    parser.add_argument('--phenotype', default="/Users/hechang/Documents/chenlab/CellHit_onlyCPU/data/metadata/clinical_TumorCompendium_v11_PolyA_for_GEO_20240520.tsv", help="Path to the phenotype file.")
    parser.add_argument('--indications_file', default="/Users/hechang/Documents/chenlab/CellHit_onlyCPU/results/feat_C2s/gdsc_clinical_indications.csv", help="Path to the standardized indications CSV file (output of standardize_indications.py).")
    parser.add_argument('--output_dir', default='./validation_per_drug_cclev2')
    parser.add_argument('--top_n', type=int, default=600)
   
    args = parser.parse_args()

    # --- 0. Setup ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Load All Data ---
    print("--- 1. Loading All Data ---")
    try:
        predictions_df = pd.read_csv(args.predictions_csv)
        print(f"Loaded {len(predictions_df)} total predictions from single file.")
        
        phenotype_df = pd.read_csv(args.phenotype, sep='\t')
        id_col_in_pheno = 'th_dataset_id'
        patient_to_disease_map = phenotype_df[[id_col_in_pheno, 'disease']].drop_duplicates()
        
        clinical_indications = load_clinical_indications_from_file(args.indications_file)
        if not clinical_indications:
            print("Clinical indications dictionary is empty. Exiting.")
            return

    except Exception as e:
        print(f"Error during data loading: {e}")
        return

    # --- 2. 准备用于分析的DataFrame ---
    print("\n--- 2. Preparing Analysis DataFrame ---")
    
    analysis_df = pd.merge(
        predictions_df,
        patient_to_disease_map,
        left_on='SampleID',
        right_on=id_col_in_pheno,
        how='left'
    ).dropna(subset=['disease'])
    
    print(f"Successfully annotated {len(analysis_df)} predictions with disease types.")
    
    # --- 3. 按药物分组并执行验证 ---
    print(f"\n--- 3. Starting Validation for {analysis_df['DrugName'].nunique()} unique drugs ---")
    
    all_results = []
    
    # [修改点 1] 初始化一个列表用于存储每个药物的真实标签信息
    ground_truth_list = [] 
    
    for drug_name, drug_df in tqdm(analysis_df.groupby('DrugName'), desc="Validating Drugs"):
        
        if drug_name not in clinical_indications or not clinical_indications[drug_name]:
            continue
        
        on_label_diseases = clinical_indications[drug_name]

        # --- 计算真实标签 (y_true) ---
        # 这里的逻辑是：如果样本的疾病在药物适应症列表中，则为1（敏感），否则为0
        drug_df['y_true'] = drug_df['disease'].isin(on_label_diseases).astype(int)
        
        # [修改点 2] 收集当前药物的 SampleID 和 y_true
        # 我们创建一个小的DataFrame片段并保存下来
        current_truth = drug_df[['SampleID', 'y_true']].copy()
        current_truth['DrugName'] = drug_name
        ground_truth_list.append(current_truth)
        
        # --- a) Top-N Enrichment Analysis ---
        drug_df_sorted = drug_df.sort_values('Predicted_IC50', ascending=True) 
        top_n_df = drug_df_sorted.head(args.top_n)

        M = len(drug_df)
        n = drug_df['y_true'].sum() # 这里直接用y_true求和，等同于原来的 drug_df['disease'].isin(on_label_diseases).sum()
        N = len(top_n_df)
        k = top_n_df['disease'].isin(on_label_diseases).sum()
        
        if n < 3: continue
        if N == 0: continue

        p_val_hypergeom = hypergeom.sf(k - 1, M, n, N)
        recall_at_top_n = k / n
        
        # --- b) AUC Analysis ---
        drug_df['y_score'] = -drug_df['Predicted_IC50']
        
        auc_score = roc_auc_score(drug_df['y_true'], drug_df['y_score']) if drug_df['y_true'].nunique() > 1 else np.nan

        # --- c) 整合结果 ---
        result = {
            'DrugName': drug_name,
            'Total TCGA Samples': M,
            'Total On-Label Samples': n,
            'Recall @ Top-N': recall_at_top_n,
            'Hypergeometric P-Value': p_val_hypergeom,
            'AUC': auc_score,
            'Is Enriched (p<0.05)': p_val_hypergeom < 0.05,
            'Is Predictive (AUC>0.5)': auc_score > 0.5
        }
        all_results.append(result)

    # --- 4. 汇总并保存所有药物的验证结果 ---
    if not all_results:
        print("\nNo validation could be performed.")
    else:
        # --- 保存验证统计结果 ---
        final_results_df = pd.DataFrame(all_results).sort_values('Hypergeometric P-Value')
        
        print("\n--- Overall Validation Summary (Enrichment & AUC) ---")
        # 只打印前几行以免刷屏
        print(final_results_df.head().to_string())
        
        enrichment_success_rate = final_results_df['Is Enriched (p<0.05)'].mean()
        auc_success_rate = final_results_df.dropna(subset=['Is Predictive (AUC>0.5)'])['Is Predictive (AUC>0.5)'].mean()
        
        print(f"\nOverall Success Rate (Enrichment, p < 0.05): {enrichment_success_rate:.2%}")
        print(f"Overall Success Rate (AUC > 0.5): {auc_success_rate:.2%}")
        
        output_csv_path = output_dir / "all_drugs_validation_summary_singlefile.csv"
        final_results_df.to_csv(output_csv_path, index=False)
        print(f"Summary results saved to: {output_csv_path}")
        
        # --- [修改点 3] 保存真实敏感性矩阵文件 ---
        if ground_truth_list:
            print("\nGenerating Ground Truth Matrix (Sample x Drug)...")
            # 1. 将列表合并为一个长表 (Long DataFrame)
            all_truth_df = pd.concat(ground_truth_list, ignore_index=True)
            
            # 2. 透视 (Pivot) 为矩阵
            # index=行(药物名), columns=列(样本名), values=值(是否敏感)
            truth_matrix = all_truth_df.pivot(index='DrugName', columns='SampleID', values='y_true')
            
            # 3. 填充 NaN (如果某些药物没有某些样本的预测，这里假设为0或者可以保留为空，通常填0表示非On-Label)
            truth_matrix = truth_matrix.fillna(0).astype(int)
            
            # 4. 保存
            matrix_output_path = output_dir / "ground_truth_sensitivity_matrix.csv"
            truth_matrix.to_csv(matrix_output_path)
            print(f"Ground truth sensitivity matrix saved to: {matrix_output_path}")

    print(f"\nValidation complete.")

if __name__ == '__main__':
    main()