"""
Script 4: Predict Drug Sensitivity using Embeddings

Predicts drug sensitivity for TCGA samples using CCLE embeddings as training data.
Supports 3 regression models: k-NN, GPR, LightGBM.

Handles two CCLE prompt formats:
  - ASPECT-2k: CCLE prompts without DrugID (top-N per cell line)
  - ASPECT-comb: CCLE prompts with DrugID/DrugName (knowledge-based per drug-cell pair)

Usage:
    # ASPECT-2k mode (CCLE top2000, no DrugID in prompts)
    python 4_predict_sensitivity.py --ccle_strategy topn --model_type knn --k_neighbors 10

    # ASPECT-comb mode (knowledge CCLE, has DrugID)
    python 4_predict_sensitivity.py --ccle_strategy knowledge --model_type knn --k_neighbors 10
"""

import numpy as np
import pandas as pd
import argparse
import time
from tqdm import tqdm
import json
from pathlib import Path
import sys

# Model imports
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import lightgbm as lgb

# Import from ASPECT (self-contained, no CellHit dependency)
from ASPECT.dataset_loaders import obtain_metadata


def load_data_topn(ccle_prompts_path, ccle_embeddings_path, 
                   tcga_prompts_path, tcga_embeddings_path,
                   dataset, data_path):
    """
    Load data for ASPECT-2k mode (CCLE prompts have no DrugID).
    Each CCLE cell line's embedding is replicated for all its drug IC50 entries.
    """
    print("--- Loading Data (ASPECT-2k) ---")
    
    ccle_prompts_df = pd.read_csv(ccle_prompts_path)
    ccle_embeddings = np.load(ccle_embeddings_path).astype(np.float32)
    tcga_prompts_df = pd.read_csv(tcga_prompts_path)
    tcga_embeddings = np.load(tcga_embeddings_path).astype(np.float32)
    
    # NaN cleanup
    for name, prompts, emb in [
        ('CCLE', ccle_prompts_df, ccle_embeddings),
        ('TCGA', tcga_prompts_df, tcga_embeddings)
    ]:
        if np.isnan(emb).any():
            mask = np.isnan(emb).any(axis=1)
            print(f"Warning: {mask.sum()} NaN rows in {name} embeddings, removing.")
            prompts.drop(prompts.index[mask], inplace=True)
            prompts.reset_index(drop=True, inplace=True)
            emb = emb[~mask]
        if name == 'CCLE':
            ccle_embeddings = emb
        else:
            tcga_embeddings = emb
    
    if len(ccle_prompts_df) != ccle_embeddings.shape[0]:
        raise ValueError(f"CCLE prompts ({len(ccle_prompts_df)}) vs embeddings ({ccle_embeddings.shape[0]}) mismatch")
    if len(tcga_prompts_df) != tcga_embeddings.shape[0]:
        raise ValueError(f"TCGA prompts ({len(tcga_prompts_df)}) vs embeddings ({tcga_embeddings.shape[0]}) mismatch")
    
    # Map CCLE SampleID -> embedding
    sample_to_emb = {
        ccle_prompts_df.iloc[i]['SampleID']: ccle_embeddings[i]
        for i in range(len(ccle_prompts_df))
    }
    
    # Load IC50 metadata and replicate embeddings per drug
    print(f"Loading '{dataset}' metadata...")
    metadata_df = obtain_metadata(dataset=dataset, path=Path(data_path))
    
    # Filter to CCLE cell lines present in prompts
    ccle_sample_ids = set(ccle_prompts_df['SampleID'])
    ccle_metadata = metadata_df[metadata_df['DepMapID'].isin(ccle_sample_ids)]
    print(f"Matched {len(ccle_metadata)} (cell, drug, IC50) entries from metadata")
    
    # Build CCLE training DataFrame
    train_data = []
    for _, row in tqdm(ccle_metadata.iterrows(), total=len(ccle_metadata), desc="Building CCLE train"):
        sid = row['DepMapID']
        if sid not in sample_to_emb:
            continue
        train_data.append({
            'SampleID': sid,
            'DrugID': row['DrugID'],
            'DrugName': row['Drug'],
            'Y': row['Y'],
            'embedding': sample_to_emb[sid]
        })
    
    ccle_df = pd.DataFrame(train_data)
    print(f"CCLE reference: {len(ccle_df)} entries, {ccle_df['DrugName'].nunique()} drugs, {ccle_df['SampleID'].nunique()} cells")
    
    # TCGA query
    tcga_df = pd.DataFrame({
        'SampleID': tcga_prompts_df['SampleID'].values,
        'embedding': list(tcga_embeddings)
    })
    print(f"TCGA query: {len(tcga_df)} samples")
    
    return ccle_df, tcga_df


def load_data_knowledge(ccle_prompts_path, ccle_embeddings_path,
                        tcga_prompts_path, tcga_embeddings_path,
                        dataset, data_path):
    """
    Load data for ASPECT-comb mode (CCLE prompts have DrugID/DrugName).
    Direct merge on ['SampleID', 'DrugID'] with IC50 data.
    """
    print("--- Loading Data (ASPECT-comb) ---")
    
    ccle_prompts_df = pd.read_csv(ccle_prompts_path)
    ccle_embeddings = np.load(ccle_embeddings_path).astype(np.float32)
    tcga_prompts_df = pd.read_csv(tcga_prompts_path)
    tcga_embeddings = np.load(tcga_embeddings_path).astype(np.float32)
    
    # NaN cleanup
    for prompts, emb in [(ccle_prompts_df, ccle_embeddings), (tcga_prompts_df, tcga_embeddings)]:
        if np.isnan(emb).any():
            mask = np.isnan(emb).any(axis=1)
            print(f"Warning: {mask.sum()} NaN rows, removing.")
            prompts.drop(prompts.index[mask], inplace=True)
            prompts.reset_index(drop=True, inplace=True)
            if prompts is ccle_prompts_df:
                ccle_embeddings = emb[~mask]
            else:
                tcga_embeddings = emb[~mask]
    
    ccle_prompts_df['embedding'] = list(ccle_embeddings)
    tcga_prompts_df['embedding'] = list(tcga_embeddings)
    
    # Merge with IC50 on ['SampleID', 'DrugID']
    print(f"Loading '{dataset}' metadata for IC50...")
    full_meta = obtain_metadata(dataset=dataset, path=Path(data_path))
    ic50_data = full_meta[['DepMapID', 'DrugID', 'Drug', 'Y']]
    
    ccle_df = ccle_prompts_df.merge(
        ic50_data,
        left_on=['SampleID', 'DrugID'],
        right_on=['DepMapID', 'DrugID'],
        how='left'
    )
    
    if ccle_df['Y'].isnull().any():
        n_missing = ccle_df['Y'].isnull().sum()
        print(f"Warning: {n_missing} entries could not be matched with IC50. Dropping.")
        ccle_df.dropna(subset=['Y'], inplace=True)
    
    print(f"CCLE reference: {len(ccle_df)} entries, {ccle_df['DrugName'].nunique()} drugs")
    print(f"TCGA query: {len(tcga_prompts_df)} samples")
    
    return ccle_df, tcga_prompts_df


def predict_for_drug(drug_name, ccle_db, tcga_queries, model_type='knn', k_neighbors=10):
    """Predict sensitivity for a single drug."""
    drug_ccle = ccle_db[ccle_db['DrugName'] == drug_name].copy().reset_index(drop=True)
    if len(drug_ccle) < 20:
        print(f"Warning: '{drug_name}' only {len(drug_ccle)} ref points. Skipping.")
        return None
    
    X_train = np.vstack(drug_ccle['embedding'].values)
    y_train = drug_ccle['Y'].values
    X_query = np.vstack(tcga_queries['embedding'].values)
    patient_ids = tcga_queries['SampleID'].values
    
    print(f"\nPredicting '{drug_name}' ({model_type.upper()}, {len(drug_ccle)} train)...")
    
    neighbor_names = None
    
    if model_type == 'knn':
        model = KNeighborsRegressor(n_neighbors=k_neighbors, weights='distance')
        model.fit(X_train, y_train)
        predictions = model.predict(X_query)
        uncertainty = np.full_like(predictions, np.nan)
        
        # Get KNN neighbors
        neighbor_indices = model.kneighbors(X_query, return_distance=False)
        train_sids = drug_ccle['SampleID'].values
        neighbor_names = [";".join(train_sids[row]) for row in neighbor_indices]
    
    elif model_type == 'gpr':
        kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        model = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=5, random_state=42)
        model.fit(X_train, y_train)
        predictions, std_dev = model.predict(X_query, return_std=True)
        uncertainty = std_dev
    
    elif model_type == 'lgbm':
        model = lgb.LGBMRegressor(random_state=42, n_estimators=100, learning_rate=0.05, num_leaves=31, verbose=-1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_query)
        uncertainty = np.full_like(predictions, np.nan)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    data = {
        'SampleID': patient_ids,
        'DrugName': drug_name,
        'Predicted_IC50': predictions,
        f'{model_type.upper()}_Uncertainty': uncertainty,
        'KNN_Neighbors': neighbor_names if neighbor_names is not None else [None] * len(patient_ids)
    }
    
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description="Predict drug sensitivity using embeddings.")
    
    # Strategy
    parser.add_argument('--ccle_strategy', type=str, default='topn',
                        choices=['topn', 'knowledge'],
                        help="CCLE prompt format: 'topn' (ASPECT-2k, no DrugID) or 'knowledge' (ASPECT-comb, has DrugID)")
    
    # File paths
    parser.add_argument('--ccle_prompts', type=str, default='./results/ccle_top2000_prompts.csv')
    parser.add_argument('--ccle_embeddings', type=str, default='./results/embeddings/ccle_embeddings.npy')
    parser.add_argument('--tcga_prompts', type=str, default='./results/tcga_top2000_prompts.csv')
    parser.add_argument('--tcga_embeddings', type=str, default='./results/embeddings/tcga_embeddings.npy')
    parser.add_argument('--output_file', type=str, default='./results/predictions.csv')
    
    # Metadata
    parser.add_argument('--dataset', type=str, default='gdsc', choices=['gdsc', 'prism'])
    parser.add_argument('--data_path', type=str, default='./data')
    
    # Model
    parser.add_argument('--model_type', type=str, default='knn', choices=['knn', 'gpr', 'lgbm'])
    parser.add_argument('--drug_names', type=str, default='["ALL"]')
    parser.add_argument('--k_neighbors', type=int, default=10)
    
    args = parser.parse_args()
    
    # Load data based on strategy
    if args.ccle_strategy == 'topn':
        ccle_df, tcga_df = load_data_topn(
            args.ccle_prompts, args.ccle_embeddings,
            args.tcga_prompts, args.tcga_embeddings,
            args.dataset, args.data_path
        )
    else:
        ccle_df, tcga_df = load_data_knowledge(
            args.ccle_prompts, args.ccle_embeddings,
            args.tcga_prompts, args.tcga_embeddings,
            args.dataset, args.data_path
        )
    
    # Determine drugs
    if '["ALL"]' in args.drug_names:
        drugs_to_predict = ccle_df['DrugName'].unique().tolist()
    else:
        drugs_to_predict = json.loads(args.drug_names)
    
    print(f"Processing {len(drugs_to_predict)} drugs with {args.model_type.upper()} (k={args.k_neighbors})")
    
    # Predict
    all_predictions = []
    for drug in tqdm(drugs_to_predict, desc="Predicting"):
        pred = predict_for_drug(drug, ccle_df, tcga_df, args.model_type, args.k_neighbors)
        if pred is not None:
            all_predictions.append(pred)
    
    if not all_predictions:
        print("No predictions made. Exiting.")
        return
    
    final = pd.concat(all_predictions, ignore_index=True)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)
    print(f"Saved {len(final)} predictions to: {output_path}")


if __name__ == '__main__':
    main()
