"""
Script 4: Predict Drug Sensitivity using Embeddings

This script predicts drug sensitivity for target samples using:
- CCLE embeddings as training data
- TCGA/custom embeddings as query data
- Multiple regression models (k-NN, GPR, LightGBM)

Usage:
    python 4_predict_sensitivity.py --model_type knn --k_neighbors 10
    python 4_predict_sensitivity.py --model_type gpr
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


def load_data(ccle_prompts_path, ccle_embeddings_path, tcga_prompts_path, tcga_embeddings_path):
    """
    Load all necessary prompt and embedding files with NaN checking.
    """
    print("--- Loading Data ---")
    
    # Load CCLE Data
    print(f"Loading CCLE prompts from: {ccle_prompts_path}")
    ccle_prompts_df = pd.read_csv(ccle_prompts_path)
    print(f"Loading CCLE embeddings from: {ccle_embeddings_path}")
    ccle_embeddings = np.load(ccle_embeddings_path).astype(np.float32)

    # Check and handle NaN in CCLE embeddings
    if np.isnan(ccle_embeddings).any():
        print(f"Warning: Found {np.isnan(ccle_embeddings).sum()} NaN values in CCLE embeddings.")
        nan_rows_mask = np.isnan(ccle_embeddings).any(axis=1)
        num_nan_rows = nan_rows_mask.sum()
        print(f"Removing {num_nan_rows} prompts/embeddings that contain NaN.")
        ccle_prompts_df = ccle_prompts_df[~nan_rows_mask].reset_index(drop=True)
        ccle_embeddings = ccle_embeddings[~nan_rows_mask]

    if len(ccle_prompts_df) != ccle_embeddings.shape[0]:
        raise ValueError("Mismatch between CCLE prompts and embeddings after cleaning.")
    
    ccle_prompts_df['embedding'] = list(ccle_embeddings)
    print(f"Loaded and cleaned {len(ccle_prompts_df)} CCLE reference points.")

    # Load TCGA Data
    print(f"Loading TCGA prompts from: {tcga_prompts_path}")
    tcga_prompts_df = pd.read_csv(tcga_prompts_path)
    print(f"Loading TCGA embeddings from: {tcga_embeddings_path}")
    tcga_embeddings = np.load(tcga_embeddings_path).astype(np.float32)

    # Check and handle NaN in TCGA embeddings
    if np.isnan(tcga_embeddings).any():
        print(f"Warning: Found {np.isnan(tcga_embeddings).sum()} NaN values in TCGA embeddings.")
        nan_rows_mask = np.isnan(tcga_embeddings).any(axis=1)
        num_nan_rows = nan_rows_mask.sum()
        print(f"Removing {num_nan_rows} prompts/embeddings that contain NaN.")
        tcga_prompts_df = tcga_prompts_df[~nan_rows_mask].reset_index(drop=True)
        tcga_embeddings = tcga_embeddings[~nan_rows_mask]

    if len(tcga_prompts_df) != tcga_embeddings.shape[0]:
        raise ValueError("Mismatch between TCGA prompts and embeddings after cleaning.")
        
    tcga_prompts_df['embedding'] = list(tcga_embeddings)
    print(f"Loaded and cleaned {len(tcga_prompts_df)} TCGA query samples.")
    
    print("-" * 40)
    return ccle_prompts_df, tcga_prompts_df


def predict_for_drug(drug_name, ccle_db, tcga_queries, model_type='knn', k_neighbors=10):
    """
    Predict sensitivity for a single drug using specified regression model.
    """
    
    # Prepare training data (CCLE)
    drug_specific_ccle = ccle_db[ccle_db['DrugName'] == drug_name].copy().reset_index(drop=True)
    if len(drug_specific_ccle) < 20:
        print(f"Warning: Drug '{drug_name}' has only {len(drug_specific_ccle)} reference points. Skipping.")
        return None
        
    X_train = np.vstack(drug_specific_ccle['embedding'].values)
    y_train = drug_specific_ccle['Y'].values
    
    # Prepare query data (TCGA)
    X_query = np.vstack(tcga_queries['embedding'].values)
    patient_ids = tcga_queries['SampleID'].values
    
    print(f"\nTraining and predicting for '{drug_name}' using {model_type.upper()} model...")
    
    model = None
    neighbor_names_column = None
    
    if model_type == 'knn':
        model = KNeighborsRegressor(n_neighbors=k_neighbors, weights='distance')
        model.fit(X_train, y_train)
        predictions = model.predict(X_query)
        uncertainty = np.full_like(predictions, np.nan)
        
        # Get KNN neighbors
        neighbor_indices = model.kneighbors(X_query, return_distance=False)
        train_sample_ids = drug_specific_ccle['SampleID'].values
        
        neighbor_names_list = []
        for row_indices in neighbor_indices:
            names = train_sample_ids[row_indices]
            neighbor_names_list.append(";".join(names))
            
        neighbor_names_column = neighbor_names_list

    elif model_type == 'gpr':
        kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        model = GaussianProcessRegressor(
            kernel=kernel, 
            alpha=0.1, 
            n_restarts_optimizer=5, 
            random_state=42
        )
        model.fit(X_train, y_train)
        predictions, std_deviation = model.predict(X_query, return_std=True)
        uncertainty = std_deviation

    elif model_type == 'lgbm':
        model = lgb.LGBMRegressor(
            random_state=42, 
            n_estimators=100, 
            learning_rate=0.05, 
            num_leaves=31
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_query)
        uncertainty = np.full_like(predictions, np.nan)

    else:
        raise ValueError("Invalid model_type. Choose from 'knn', 'gpr', 'lgbm'.")

    # Build output DataFrame
    data_dict = {
        'SampleID': patient_ids,
        'DrugName': drug_name,
        'Predicted_IC50': predictions,
        f'{model_type.upper()}_Uncertainty': uncertainty
    }
    
    if model_type == 'knn' and neighbor_names_column is not None:
        data_dict['KNN_Neighbors'] = neighbor_names_column
    else:
        data_dict['KNN_Neighbors'] = [None] * len(patient_ids)

    results_df = pd.DataFrame(data_dict)
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Predict drug sensitivity for TCGA patients using embeddings."
    )
    
    # Path parameters
    parser.add_argument(
        '--ccle_prompts', 
        type=str, 
        default='./results/gdsc_ccle_mechanism_prompts.csv',
        help="Path to CCLE prompts CSV"
    )
    parser.add_argument(
        '--ccle_embeddings', 
        type=str,
        default='./results/embeddings/ccle_embeddings.npy',
        help="Path to CCLE embeddings NPY"
    )
    parser.add_argument(
        '--tcga_prompts', 
        type=str, 
        default='./results/tcga_top2000_prompts.csv',
        help="Path to TCGA prompts CSV"
    )
    parser.add_argument(
        '--tcga_embeddings', 
        type=str, 
        default='./results/embeddings/tcga_embeddings.npy',
        help="Path to TCGA embeddings NPY"
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='./results/predictions.csv',
        help="Path to save predictions"
    )
    
    # Data loading parameters
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='gdsc', 
        choices=['gdsc', 'prism'],
        help="Dataset name for metadata loading"
    )
    parser.add_argument(
        '--data_path', 
        type=str, 
        default='./data',
        help="Root data directory"
    )
 
    # Model selection
    parser.add_argument(
        '--model_type', 
        type=str, 
        default='knn', 
        choices=['knn', 'gpr', 'lgbm'],
        help="Regression model to use"
    )

    # Prediction parameters
    parser.add_argument(
        '--drug_names', 
        type=str, 
        default='["ALL"]',
        help="JSON list of drug names to predict, or '[\"ALL\"]' for all"
    )
    parser.add_argument(
        '--k_neighbors', 
        type=int, 
        default=10, 
        help="Number of neighbors for k-NN model"
    )
    
    args = parser.parse_args()

    # Load data
    ccle_df, tcga_df = load_data(
        args.ccle_prompts, 
        args.ccle_embeddings, 
        args.tcga_prompts, 
        args.tcga_embeddings
    )

    # Load IC50 data
    print(f"Loading '{args.dataset}' metadata for IC50 values...")
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
        print(f"Warning: {ccle_df['Y'].isnull().sum()} CCLE prompts could not be matched with IC50.")
        ccle_df.dropna(subset=['Y'], inplace=True)
        print(f"Removed unmatched entries. {len(ccle_df)} CCLE reference points remain.")

    # Determine drugs to process
    if '["ALL"]' in args.drug_names:
        drugs_to_predict = ccle_df['DrugName'].unique().tolist()
    else:
        drugs_to_predict = json.loads(args.drug_names)
    
    print(f"Will process {len(drugs_to_predict)} drugs using model: {args.model_type.upper()}")

    # Run predictions
    all_predictions = []
    
    for drug in tqdm(drugs_to_predict, desc="Processing Drugs"):
        predictions_df = predict_for_drug(
            drug, ccle_df, tcga_df, 
            model_type=args.model_type, 
            k_neighbors=args.k_neighbors
        )
        if predictions_df is not None:
            all_predictions.append(predictions_df)
            
    # Save results
    if not all_predictions:
        print("No predictions were made. Exiting.")
        return
        
    final_predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    print(f"\n--- Saving Final Predictions ---")
    
    # Create output directory if needed
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    final_predictions_df.to_csv(args.output_file, index=False)
    print(f"Results saved to: {args.output_file}")


if __name__ == '__main__':
    main()
