# predict_sensitivity.py (Advanced Regression Models Version)

import numpy as np
import pandas as pd
import argparse
import time
from tqdm import tqdm
import json
from pathlib import Path
import sys

# --- 模型导入 ---
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import lightgbm as lgb

# --- 从您的工具包中导入 obtain_metadata ---
# (确保路径正确)
#sys.path.append('./../../') 
from CellHit.data import obtain_metadata

def load_data(ccle_prompts_path, ccle_embeddings_path, tcga_prompts_path, tcga_embeddings_path):
    """
    Load all necessary prompt and embedding files, with NaN checking and cleaning.
    """
    print("--- Loading Data ---")
    
    # --- Load CCLE Data ---
    print(f"Loading CCLE prompts from: {ccle_prompts_path}")
    ccle_prompts_df = pd.read_csv(ccle_prompts_path)
    print(f"Loading CCLE embeddings from: {ccle_embeddings_path}")
    ccle_embeddings = np.load(ccle_embeddings_path).astype(np.float32) # 确保是float32

    # --- 核心修改：检查并处理NaN ---
    if np.isnan(ccle_embeddings).any():
        print(f"Warning: Found {np.isnan(ccle_embeddings).sum()} NaN values in CCLE embeddings.")
        # 找到包含NaN的行的索引
        nan_rows_mask = np.isnan(ccle_embeddings).any(axis=1)
        num_nan_rows = nan_rows_mask.sum()
        print(f"Removing {num_nan_rows} prompts/embeddings that contain NaN.")
        # 从 prompts 和 embeddings 中同时移除这些行
        ccle_prompts_df = ccle_prompts_df[~nan_rows_mask].reset_index(drop=True)
        ccle_embeddings = ccle_embeddings[~nan_rows_mask]

    # Sanity check
    if len(ccle_prompts_df) != ccle_embeddings.shape[0]:
        raise ValueError("Mismatch between number of CCLE prompts and embeddings after cleaning.")
    
    ccle_prompts_df['embedding'] = list(ccle_embeddings)
    print(f"Loaded and cleaned {len(ccle_prompts_df)} CCLE reference points.")

    # --- Load TCGA Data (同样处理) ---
    print(f"Loading TCGA prompts from: {tcga_prompts_path}")
    tcga_prompts_df = pd.read_csv(tcga_prompts_path)
    print(f"Loading TCGA embeddings from: {tcga_embeddings_path}")
    tcga_embeddings = np.load(tcga_embeddings_path).astype(np.float32)

    # --- 核心修改：检查并处理NaN ---
    if np.isnan(tcga_embeddings).any():
        print(f"Warning: Found {np.isnan(tcga_embeddings).sum()} NaN values in TCGA embeddings.")
        nan_rows_mask = np.isnan(tcga_embeddings).any(axis=1)
        num_nan_rows = nan_rows_mask.sum()
        print(f"Removing {num_nan_rows} prompts/embeddings that contain NaN.")
        tcga_prompts_df = tcga_prompts_df[~nan_rows_mask].reset_index(drop=True)
        tcga_embeddings = tcga_embeddings[~nan_rows_mask]

    if len(tcga_prompts_df) != tcga_embeddings.shape[0]:
        raise ValueError("Mismatch between number of TCGA prompts and embeddings after cleaning.")
        
    tcga_prompts_df['embedding'] = list(tcga_embeddings)
    print(f"Loaded and cleaned {len(tcga_prompts_df)} TCGA query samples.")
    
    print("-" * 40)
    return ccle_prompts_df, tcga_prompts_df

# --- 核心修改：预测逻辑 ---
def predict_for_drug_advanced(drug_name, ccle_db, tcga_queries, model_type='gpr', k_neighbors=10):
    """
    Predict sensitivity for a single drug using advanced regression models.
    """
    
    # 1. 准备训练数据 (CCLE)
    drug_specific_ccle = ccle_db[ccle_db['DrugName'] == drug_name].copy().reset_index(drop=True) # ### 修改1: reset_index 确保索引与numpy数组对齐
    if len(drug_specific_ccle) < 20: 
        print(f"Warning: Drug '{drug_name}' has only {len(drug_specific_ccle)} reference points. Skipping for advanced model.")
        return None
        
    X_train = np.vstack(drug_specific_ccle['embedding'].values)
    y_train = drug_specific_ccle['Y'].values
    
    # 2. 准备查询数据 (TCGA)
    X_query = np.vstack(tcga_queries['embedding'].values)
    patient_ids = tcga_queries['SampleID'].values
    
    # --- 3. 选择、训练和预测模型 ---
    print(f"\nTraining and predicting for '{drug_name}' using {model_type.upper()} model...")
    
    model = None
    neighbor_names_column = None # ### 修改2: 初始化变量用于存储邻居名字
    
    if model_type == 'knn':
        # 使用scikit-learn的kNN回归器
        model = KNeighborsRegressor(n_neighbors=k_neighbors, weights='distance')
        model.fit(X_train, y_train)
        predictions = model.predict(X_query)
        uncertainty = np.full_like(predictions, np.nan) 
        
        # ### 修改3: 获取 KNN 的邻居索引并转换为名字 ###
        # return_distance=False 只返回索引
        neighbor_indices = model.kneighbors(X_query, return_distance=False)
        
        # 获取训练集(CCLE)的SampleID列表
        train_sample_ids = drug_specific_ccle['SampleID'].values
        
        # 将索引转换为名字，并用分号连接
        neighbor_names_list = []
        for row_indices in neighbor_indices:
            # 通过索引找到对应的 SampleID
            names = train_sample_ids[row_indices]
            # 拼接成字符串 "ID1;ID2;ID3..."
            neighbor_names_list.append(";".join(names))
            
        neighbor_names_column = neighbor_names_list 
        # #########################################

    elif model_type == 'gpr':
        kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        model = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=5, random_state=42)
        model.fit(X_train, y_train)
        predictions, std_deviation = model.predict(X_query, return_std=True)
        uncertainty = std_deviation

    elif model_type == 'lgbm':
        model = lgb.LGBMRegressor(random_state=42, n_estimators=100, learning_rate=0.05, num_leaves=31)
        model.fit(X_train, y_train)
        predictions = model.predict(X_query)
        uncertainty = np.full_like(predictions, np.nan)

    else:
        raise ValueError("Invalid model_type. Choose from 'knn', 'gpr', 'lgbm'.")

    # 4. 整理输出
    # ### 修改4: 构建 DataFrame 时加入邻居列 ###
    data_dict = {
        'SampleID': patient_ids,
        'DrugName': drug_name,
        'Predicted_IC50': predictions,
        f'{model_type.upper()}_Uncertainty': uncertainty
    }
    
    # 如果是 KNN 模型，则添加 neighbors 列
    if model_type == 'knn' and neighbor_names_column is not None:
        data_dict['KNN_Neighbors'] = neighbor_names_column
    else:
        # 为了保持格式统一（可选），其他模型可以填 None
        data_dict['KNN_Neighbors'] = [None] * len(patient_ids)

    results_df = pd.DataFrame(data_dict)
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description="Predict drug sensitivity for TCGA patients using embeddings.")
    
 # --- 路径参数 ---
    parser.add_argument('--ccle_prompts', type=str, default='/Users/hechang/Documents/chenlab/ccle_embeddings/gdsc_ccle_mechanism_prompts_filtered_v2.csv')
    parser.add_argument('--ccle_embeddings', type=str,default='/Users/hechang/Documents/chenlab/ccle_embeddings/embeddings_filtered_v2.npy')
    parser.add_argument('--tcga_prompts', type=str, default='/Users/hechang/Documents/chenlab/tcga_embeddings/tcga_top2000_prompts_filtered.csv')
    parser.add_argument('--tcga_embeddings', type=str, default='/Users/hechang/Documents/chenlab/tcga_embeddings/all_embeddings_merged.npy')
    parser.add_argument('--output_file', type=str, default='./predictions_comb.csv')
    
    # --- 数据加载参数 (替代 --full_metadata) ---
    parser.add_argument('--dataset', type=str, default='gdsc', choices=['gdsc', 'prism'], help="Dataset name to load metadata for.")
    parser.add_argument('--data_path', type=str, default='/Users/hechang/Documents/chenlab/CellHit_onlyCPU/data', help="Root data directory for obtain_metadata.")
 
    # --- 核心修改：增加模型选择参数 ---
    parser.add_argument(
        '--model_type', 
        type=str, 
        default='knn', 
        choices=['knn', 'gpr', 'lgbm'],
        help="The regression model to use in the embedding space."
    )

    # --- 预测参数 ---
    parser.add_argument('--drug_names', type=str, default='[\"ALL\"]')
    parser.add_argument('--k_neighbors', type=int, default=10, help="Number of neighbors for k-NN model.")
    
    args = parser.parse_args()

    # --- 1. 加载数据 ---
    ccle_df, tcga_df = load_data(args.ccle_prompts, args.ccle_embeddings, args.tcga_prompts, args.tcga_embeddings)

    # --- 2. 加载IC50 ---
    print(f"Dynamically loading '{args.dataset}' metadata to get IC50 values...")
    full_meta_df = obtain_metadata(dataset=args.dataset, path=Path(args.data_path))
    ic50_data = full_meta_df[['DepMapID', 'DrugID', 'Drug', 'Y']]
    
    ccle_df = pd.merge(
        ccle_df,
        ic50_data,
        left_on=['SampleID', 'DrugID'],
        right_on=['DepMapID', 'DrugID'],
        how='left'
    )
    
    if ccle_df['Y'].isnull().any():
        print(f"Warning: {ccle_df['Y'].isnull().sum()} CCLE prompts could not be matched with an IC50 value.")
        ccle_df.dropna(subset=['Y'], inplace=True)
        print(f"Removed unmatched entries. {len(ccle_df)} CCLE reference points remain.")

    # --- 3. 确定处理药物列表 ---
    if '["ALL"]' in args.drug_names:
        drugs_to_predict = ccle_df['DrugName'].unique().tolist()
    else:
        drugs_to_predict = json.loads(args.drug_names)
    print(f"Will process {len(drugs_to_predict)} drugs using model: {args.model_type.upper()}")

    # --- 4. 运行预测 ---
    all_predictions = []
    
    for drug in tqdm(drugs_to_predict, desc="Processing Drugs"):
        # 调用新的预测函数
        predictions_df = predict_for_drug_advanced(
            drug, ccle_df, tcga_df, model_type=args.model_type, k_neighbors=args.k_neighbors
        )
        if predictions_df is not None:
            all_predictions.append(predictions_df)
            
    # --- 5. 汇总并保存 ---
    if not all_predictions:
        print("No predictions were made. Exiting.")
        return
        
    final_predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    print(f"\n--- Saving Final Predictions ---")
    final_predictions_df.to_csv(args.output_file, index=False)
    print(f"Results saved to: {args.output_file}")


if __name__ == '__main__':
    main()