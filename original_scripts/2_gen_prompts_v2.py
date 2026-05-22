# gen_prompt.py (Final Hybrid Strategy with Filtering - Updated)

import pandas as pd
from pathlib import Path
import argparse
import time
import json
import sys
from tqdm import tqdm

# 从您的工具包中导入需要的工具函数
# 注意：如果 dataset 是 others，可能不需要 obtain_metadata，这里做个 try-import 或者在调用时控制
from moa_c2s.dataset_loaders import obtain_metadata
from moa_c2s.gen_gene_list import GeneGetter

def load_custom_csv(csv_path):
    """
    加载自定义 CSV 文件。
    假设输入格式：行 = Genes, 列 = Samples
    输出格式：行 = Samples, 列 = Genes (Pandas 标准格式)
    """
    print(f"Loading custom expression CSV from: {csv_path}")
    try:
        # index_col=0 假设第一列是基因名
        df = pd.read_csv(csv_path, index_col=0)
        # 转置：将 (Genes x Samples) 转换为 (Samples x Genes)
        df = df.T
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
    
    # --- 参数定义 ---
    # 修改 1: 加入 'others' 选项
    parser.add_argument('--dataset', type=str, required=True, choices=['gdsc', 'prism', 'others'])
    
    parser.add_argument('--data_path', type=str, default='/home/hechang/merged_frame/data')
    parser.add_argument('--celligner_path', type=str, default='/home/hechang/merged_frame/data/transcriptomics/celligner_CCLE_TCGA.feather')
    
    # 修改 2: 新增 CSV 输入参数
    parser.add_argument('--expression_csv', type=str, default=None, 
                        help="Path to custom expression CSV (Rows=Genes, Cols=Samples). If provided, this data replaces the TCGA/Target part.")
    
    parser.add_argument('--output_path', type=str, default='/home/hechang/merged_frame/results')
    parser.add_argument('--top_n_genes', type=int, default=2000, help="Number of top expressed genes for TCGA/Custom samples.")
    
    # --- 筛选参数 ---
    parser.add_argument('--filter_projects', type=str, default=None, help="JSON list of TCGA project_ids to keep (Only applicable for standard TCGA data).")
    parser.add_argument('--filter_drug_names', type=str, default=None, help="JSON list of Drug Names to apply globally (Only applicable for GDSC/PRISM).")
    
    args = parser.parse_args()

    start_time = time.time()
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Starting Hybrid Prompt Generation (Dataset: {args.dataset}) ---")

    # --- 1. 数据加载逻辑 (核心修改) ---
    
    ccle_expr = pd.DataFrame()
    tcga_expr = pd.DataFrame() # 这里泛指所有 Target/Inference 样本
    
    if args.dataset == 'others':
        # 模式 A: Others
        # 必须提供 CSV，不加载 Celligner Feather，不加载 Drug Metadata
        if not args.expression_csv:
            print("Error: For dataset 'others', you must provide --expression_csv.")
            sys.exit(1)
        
        tcga_expr = load_custom_csv(args.expression_csv)
        print("Dataset is 'others': Skipping CCLE/Mechanism loading. Only generating Top-N prompts for custom data.")
        
    else:
        # 模式 B: GDSC / PRISM
        # 1. 尝试加载 Celligner Feather (用于 CCLE 部分)
        print(f"Loading transcriptomics matrix from: {args.celligner_path}")
        try:
            transcriptomics_df = pd.read_feather(args.celligner_path).set_index('index')
            source_map = transcriptomics_df[['Source']]
            transcriptomics_df = transcriptomics_df.drop(columns=['Source'])
            
            # 提取 CCLE 用于机制 Prompt
            ccle_expr = transcriptomics_df[source_map['Source'] == 'CCLE']
            
            # 默认情况下，TCGA 部分也来自 Feather
            tcga_expr_default = transcriptomics_df[source_map['Source'] == 'TCGA']
        except Exception as e:
            print(f"Warning: Could not load Feather file: {e}")
            if not args.expression_csv:
                print("Error: No Feather file and no CSV provided. Cannot proceed.")
                sys.exit(1)

        # 2. 决定 Target 数据来源 (TCGA 或 自定义 CSV)
        if args.expression_csv:
            print("Custom CSV provided. Using CSV data as Target samples (replacing standard TCGA).")
            tcga_expr = load_custom_csv(args.expression_csv)
        else:
            print("Using standard TCGA data from Feather file.")
            tcga_expr = tcga_expr_default

        print(f"Loaded {len(ccle_expr)} CCLE samples (for Mechanism) and {len(tcga_expr)} Target samples (for Top-N).")

    # --- 2. 加载元数据与筛选 (仅针对 GDSC/PRISM) ---
    
    metadata_df = pd.DataFrame()
    drug_to_genes_ccle = {}

    if args.dataset != 'others':
        print("Loading full metadata...")
        metadata_df = obtain_metadata(dataset=args.dataset, path=Path(args.data_path))
        
        # a) 筛选药物
        if args.filter_drug_names:
            try:
                drugs_to_keep = json.loads(args.filter_drug_names)
                print(f"Filtering globally for drugs: {drugs_to_keep}")
                metadata_df = metadata_df[metadata_df['Drug'].isin(drugs_to_keep)]
                print(f"After filtering, {len(metadata_df)} total metadata pairs remain.")
            except Exception as e:
                print(f"Error parsing drug filter: {e}")

        # 准备 GeneGetter
        gene_getter = GeneGetter(
            dataset=args.dataset,
            data_path=args.data_path,
            available_genes=ccle_expr.columns # 使用 CCLE 的基因作为参考
        )
        
        # 预处理 CCLE 药物基因映射
        # 从已被药物筛选过的 metadata_df 中，再筛选出与CCLE相关的部分
        ccle_metadata = metadata_df[metadata_df['DepMapID'].isin(ccle_expr.index)]
        unique_drug_ids_ccle = ccle_metadata['DrugID'].unique()
        drug_to_genes_ccle = {drug_id: gene_getter.get_genes(drug_id) for drug_id in unique_drug_ids_ccle}
    
    # b) 筛选 TCGA 样本 (仅当不是自定义CSV且提供了筛选参数时)
    if args.filter_projects and args.dataset != 'others' and not args.expression_csv:
        try:
            projects_to_keep = json.loads(args.filter_projects)
            print(f"Filtering TCGA samples for projects: {projects_to_keep}")
            
            tcga_meta_path = Path(args.data_path) / 'metadata' / 'tcga_clinical.tsv'
            if tcga_meta_path.exists():
                tcga_meta = pd.read_csv(tcga_meta_path, sep='\t', usecols=['case_submitter_id', 'project_id'])
                id_to_project_map = tcga_meta.set_index('case_submitter_id')['project_id']
                
                # 临时添加 project_id 列进行筛选
                # 注意：Feather index 通常与 case_submitter_id 匹配
                tcga_expr_temp = tcga_expr.copy()
                tcga_expr_temp['project_id'] = tcga_expr_temp.index.map(id_to_project_map)
                tcga_expr = tcga_expr_temp[tcga_expr_temp['project_id'].isin(projects_to_keep)]
                tcga_expr = tcga_expr.drop(columns=['project_id']) # 恢复纯表达矩阵
                print(f"After filtering, {len(tcga_expr)} TCGA samples remain.")
            else:
                print(f"Warning: Metadata file not found at {tcga_meta_path}. Skipping project filtering.")
        except Exception as e:
            print(f"Warning: Could not filter TCGA by project. Error: {e}")
    elif args.filter_projects and (args.dataset == 'others' or args.expression_csv):
        print("Warning: --filter_projects is ignored when using 'others' dataset or custom CSV input.")

    # --- 3. 生成 Prompts ---
    
    # --- 3.1 处理 CCLE Prompts (机制驱动) - 仅限 GDSC/PRISM ---
    if args.dataset != 'others' and not ccle_expr.empty:
        print("\n--- Generating Mechanism-based Prompts for CCLE ---")
        
        ccle_prompts_data = []
        # 重新获取筛选后的 ccle_metadata (因为前面可能只用了 drug id)
        ccle_metadata = metadata_df[metadata_df['DepMapID'].isin(ccle_expr.index)]

        for _, row in tqdm(ccle_metadata.iterrows(), total=len(ccle_metadata), desc="Filtered CCLE Pairs"):
            sample_id = row['DepMapID']
            drug_id = row['DrugID']
            
            mechanism_genes = drug_to_genes_ccle.get(drug_id)
            if not mechanism_genes: continue
            
            # 确保基因在表达矩阵中存在
            valid_genes = [g for g in mechanism_genes if g in ccle_expr.columns]
            if not valid_genes: continue

            expression_profile = ccle_expr.loc[sample_id]
            mechanism_expression = expression_profile[valid_genes]
            sorted_genes = mechanism_expression.sort_values(ascending=False)
            sentence = " ".join(sorted_genes.index.tolist())
            
            ccle_prompts_data.append({
                'SampleID': sample_id, 'DrugID': drug_id,
                'DrugName': row['Drug'], 'Prompt_Type': 'Mechanism',
                'Prompt': sentence
            })

        # 保存 CCLE 结果
        if ccle_prompts_data:
            ccle_prompts_df = pd.DataFrame(ccle_prompts_data)
            print(f"\nGenerated {len(ccle_prompts_df)} mechanism-based prompts for CCLE.")
            ccle_output_path = output_dir / f'{args.dataset}_ccle_mechanism_prompts_filtered.csv'
            ccle_prompts_df.to_csv(ccle_output_path, index=False)
            print(f"CCLE prompts saved to: {ccle_output_path}")
        else:
            print("\nNo CCLE prompts were generated.")

    # --- 3.2 处理 Target/TCGA Prompts (Top-N驱动) ---
    # 这一步对所有模式都适用，只要 tcga_expr 不为空
    if not tcga_expr.empty:
        dataset_label = "CustomCSV" if args.expression_csv or args.dataset == 'others' else "TCGA"
        print(f"\n--- Generating Top-{args.top_n_genes} Gene Prompts for {dataset_label} ---")
        
        tcga_prompts_data = []
        
        for sample_id, expression_profile in tqdm(tcga_expr.iterrows(), total=len(tcga_expr), desc=f"Processing {dataset_label} Samples"):
            # 获取 Top N 基因
            # 如果基因数少于 Top N，则取所有
            n = min(args.top_n_genes, len(expression_profile))
            top_n_genes = expression_profile.nlargest(n)
            sentence = " ".join(top_n_genes.index.tolist())
            
            tcga_prompts_data.append({
                'SampleID': sample_id, 'DrugID': 'N/A',
                'DrugName': 'N/A', 'Prompt_Type': f'Top_{n}',
                'Prompt': sentence
            })

        # 保存 Target 结果
        if tcga_prompts_data:
            tcga_prompts_df = pd.DataFrame(tcga_prompts_data)
            print(f"Generated {len(tcga_prompts_df)} top-N-based prompts for {dataset_label}.")
            
            if args.dataset == 'others':
                 # 如果是 others，文件名通用一点
                 target_output_filename = f'custom_top{args.top_n_genes}_prompts.csv'
            elif args.expression_csv:
                 # 如果是 GDSC/PRISM 但用了自定义 CSV
                 target_output_filename = f'{args.dataset}_custom_target_top{args.top_n_genes}_prompts.csv'
            else:
                 # 标准 TCGA
                 target_output_filename = f'tcga_top{args.top_n_genes}_prompts_filtered.csv'

            tcga_output_path = output_dir / target_output_filename
            tcga_prompts_df.to_csv(tcga_output_path, index=False)
            print(f"{dataset_label} prompts saved to: {tcga_output_path}")
        else:
            print(f"\nNo {dataset_label} prompts were generated.")
    
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds.")


if __name__ == '__main__':
    main()