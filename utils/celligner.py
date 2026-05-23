"""
Simplified Celligner implementation for CCLE-TCGA alignment.

This module provides a simplified version of the Celligner alignment algorithm
for aligning cancer cell line (CCLE) and tumor (TCGA) transcriptomic data.

Note: This is a simplified implementation using PCA-based alignment.
For production use, consider installing the full celligner package:
    pip install celligner
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from pathlib import Path
import pickle


class SimpleCelligner:
    """
    A simplified Celligner for aligning CCLE and TCGA transcriptomic data.
    
    This implementation uses:
    1. PCA for dimensionality reduction
    2. K-Nearest Neighbors for cross-domain mapping
    3. Inverse transformation for final alignment
    
    Parameters
    ----------
    n_components : int, default=100
        Number of PCA components
    n_neighbors : int, default=10
        Number of neighbors for KNN alignment
    random_state : int, default=42
        Random seed for reproducibility
    """
    
    def __init__(self, n_components=100, n_neighbors=10, random_state=42):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.ccle_scaler = None
        self.tcga_scaler = None
        self.ccle_pca = None
        self.aligned_tcga = None
        self.combined_output = None
        
    def fit(self, ccle_data):
        """
        Fit the model on CCLE (reference) data.
        
        Parameters
        ----------
        ccle_data : pd.DataFrame
            CCLE expression matrix (samples x genes)
        """
        # Scale CCLE data
        self.ccle_scaler = StandardScaler()
        ccle_scaled = self.ccle_scaler.fit_transform(ccle_data)
        
        # Fit PCA on CCLE
        self.ccle_pca = PCA(n_components=self.n_components, random_state=self.random_state)
        ccle_pca = self.ccle_pca.fit_transform(ccle_scaled)
        
        # Store CCLE PCA representation
        self.ccle_pca_data_ = pd.DataFrame(
            ccle_pca, 
            index=ccle_data.index,
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )
        
        return self
    
    def transform(self, tumor_data):
        """
        Transform TCGA/tumor data to CCLE space using KNN alignment.
        
        Parameters
        ----------
        tumor_data : pd.DataFrame
            TCGA/tumor expression matrix (samples x genes)
            
        Returns
        -------
        pd.DataFrame
            TCGA data aligned to CCLE space
        """
        if self.ccle_scaler is None:
            raise ValueError("Model must be fitted before transform")
        
        # Scale tumor data using CCLE scaler
        tumor_scaled = self.ccle_scaler.transform(tumor_data)
        
        # Project tumor data to CCLE PCA space
        tumor_pca = self.ccle_pca.transform(tumor_scaled)
        tumor_pca_df = pd.DataFrame(
            tumor_pca,
            index=tumor_data.index,
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )
        
        # Use KNN to align tumor PCA to CCLE PCA
        knn = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights='distance')
        knn.fit(self.ccle_pca_data_.values, self.ccle_pca_data_.index)
        
        # Predict CCLE indices for each tumor sample
        aligned_indices = knn.predict(tumor_pca)
        
        # Compute weighted average of CCLE samples for alignment
        distances, indices = knn.kneighbors(tumor_pca)
        weights = 1 / (distances + 1e-10)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Get aligned CCLE samples
        ccle_aligned = self.ccle_scaler.inverse_transform(
            self.ccle_pca.inverse_transform(tumor_pca)
        )
        
        self.aligned_tcga = pd.DataFrame(
            ccle_aligned,
            index=tumor_data.index,
            columns=tumor_data.columns
        )
        
        return self.aligned_tcga
    
    def fit_transform(self, ccle_data, tumor_data):
        """
        Fit on CCLE and transform tumor data.
        
        Parameters
        ----------
        ccle_data : pd.DataFrame
            CCLE expression matrix
        tumor_data : pd.DataFrame
            TCGA/tumor expression matrix
            
        Returns
        -------
        pd.DataFrame
            Combined aligned data
        """
        self.fit(ccle_data)
        self.transform(tumor_data)
        return self.combined_output
    
    @property
    def combined_output(self):
        """Get combined CCLE and aligned TCGA data."""
        if self.aligned_tcga is None:
            return None
        
        # Inverse transform CCLE PCA to original space
        ccle_aligned = self.ccle_scaler.inverse_transform(
            self.ccle_pca.inverse_transform(self.ccle_pca_data_.values)
        )
        ccle_aligned_df = pd.DataFrame(
            ccle_aligned,
            index=self.ccle_pca_data_.index,
            columns=self.ccle_scaler.feature_names_in_
        )
        
        # Combine
        return pd.concat([ccle_aligned_df, self.aligned_tcga])
    
    def save(self, path):
        """
        Save the model to a pickle file.
        
        Parameters
        ----------
        path : str or Path
            Path to save the model
        """
        path = Path(path)
        with open(path, 'wb') as f:
            pickle.dump({
                'ccle_scaler': self.ccle_scaler,
                'ccle_pca': self.ccle_pca,
                'ccle_pca_data': self.ccle_pca_data_,
                'n_components': self.n_components,
                'n_neighbors': self.n_neighbors,
                'random_state': self.random_state
            }, f)
    
    @classmethod
    def load(cls, path):
        """
        Load a model from a pickle file.
        
        Parameters
        ----------
        path : str or Path
            Path to the saved model
            
        Returns
        -------
        SimpleCelligner
            Loaded model instance
        """
        path = Path(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(
            n_components=data['n_components'],
            n_neighbors=data['n_neighbors'],
            random_state=data['random_state']
        )
        model.ccle_scaler = data['ccle_scaler']
        model.ccle_pca = data['ccle_pca']
        model.ccle_pca_data_ = data['ccle_pca_data']
        
        return model


def source_mapper(x, ccle_index, tcga_index, external=None):
    """
    Map sample IDs to their source (CCLE, TCGA, or external).
    
    Parameters
    ----------
    x : str
        Sample ID
    ccle_index : pd.Index
        CCLE sample IDs
    tcga_index : pd.Index
        TCGA sample IDs
    external : str, optional
        Label for external samples not in CCLE or TCGA
        
    Returns
    -------
    str
        Source label
    """
    if x in set(ccle_index):
        return 'CCLE'
    elif x in set(tcga_index):
        return 'TCGA'
    else:
        return external
