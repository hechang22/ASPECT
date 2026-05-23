"""
Dataset loaders for MOA-C2S framework.

This module provides functions and classes for loading and processing
genomic/transcriptomic data from various sources (GDSC, PRISM, CCLE, TCGA).
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class IndexedArray:
    """
    A simple indexed array class for efficient row access by key.
    
    Similar functionality to CellHit's IndexedArray.
    """

    def __init__(self, input_dict):
        """
        Initialize IndexedArray from a dictionary.
        
        Args:
            input_dict: Dictionary mapping keys to numpy arrays
        """
        self.key_to_index = {}
        array_list = []

        for idx, (key, array) in enumerate(input_dict.items()):
            self.key_to_index[key] = idx
            array_list.append(np.array(array, dtype=float).reshape(1, -1))

        self.array = np.vstack(array_list)

    def __getitem__(self, key):
        """
        Get rows by key(s).
        
        Args:
            key: String, list of strings, or numpy array of keys
            
        Returns:
            Corresponding rows as numpy array
        """
        # if string, return the corresponding row
        if isinstance(key, str):
            return self.array[self.key_to_index[key]]
        # if list of strings, return the corresponding rows
        elif isinstance(key, list):
            return self.array[[self.key_to_index[name] for name in key]]
        # if numpy array, send to list of strings and then return the corresponding rows
        elif isinstance(key, np.ndarray):
            return self.array[[self.key_to_index[str(name)] for name in key]]
        else:
            raise TypeError('Invalid argument type.')
        
    def get_all_keys(self):
        """Return all keys in the array."""
        return list(self.key_to_index.keys())


def obtain_gdsc(data_path, drug_threshold=10, **kwargs):
    """
    Load GDSC (Genomics of Drug Sensitivity in Cancer) dataset.
    
    Args:
        data_path: Path to the data directory
        drug_threshold: Minimum number of cell lines per drug
        
    Returns:
        DataFrame with columns: COSMICID, DrugID, Drug, Y (IC50 values)
    """
    path = Path(data_path)

    # read responses from GDSC dataset
    data = pd.read_csv(path / 'metadata' / 'GDSC2_fitted_dose_response_24Jul22.csv')
    data = data.rename(columns={
        'COSMIC_ID': 'COSMICID',
        'DRUG_ID': 'DrugID',
        'DRUG_NAME': 'Drug',
        'LN_IC50': 'Y'
    })
    data = data[['COSMICID', 'DrugID', 'Drug', 'Y']]

    # read cell lines metadata
    cell_lines = pd.read_csv(path / 'metadata' / 'Model.csv')
    cell_lines = cell_lines[['ModelID', 'COSMICID', 'OncotreeCode', 'OncotreeSubtype', 
                             'OncotreePrimaryDisease', 'OncotreeLineage']]

    # merge the two pieces of information
    data = data.merge(cell_lines, on='COSMICID', how='inner').rename(columns={'ModelID': 'DepMapID'})
    
    # filter on drugs with id in GDSC_drugs.csv
    drugs = set(pd.read_csv(path / 'metadata' / 'GDSC_drugs.csv')['DRUG_ID'].values)
    data = data[data['DrugID'].isin(drugs)]

    count_drug = data['DrugID'].value_counts()
    count_drug = set(count_drug[count_drug > drug_threshold].index.tolist())
    data = data[data['DrugID'].isin(count_drug)]
    
    return data


def obtain_prism_lfc(data_path, drug_threshold=10, **kwargs):
    """
    Load PRISM (Profiling Relative Inhibition Simultaneously in Mixtures) dataset.
    
    Args:
        data_path: Path to the data directory
        drug_threshold: Minimum number of cell lines per drug
        
    Returns:
        DataFrame with drug sensitivity data
    """
    path = Path(data_path)
    
    model = pd.read_csv(path / 'metadata' / 'Model.csv')
    model = model[['ModelID', 'COSMICID', 'OncotreeCode', 'OncotreeSubtype', 
                  'OncotreePrimaryDisease', 'OncotreeLineage']]
    model = model.rename(columns={'ModelID': 'DepMapID'})
    
    prism_lfc = pd.read_csv(path / 'metadata' / 'Repurposing_Public_23Q2_Extended_Primary_Data_Matrix.csv')
    mapping_metadata = pd.read_csv(path / 'metadata' / 'Repurposing_Public_23Q2_Extended_Primary_Compound_List.csv')
    mapping_metadata = mapping_metadata[['Drug.Name', 'IDs']]
    mapping_metadata.columns = ['Drug', 'DrugID']
    
    prism_lfc.columns = ['DrugID'] + [i for i in prism_lfc.columns[1:]]
    prism_lfc = pd.merge(prism_lfc, mapping_metadata, on='DrugID', how='inner')
    # melt prism_lfc
    prism_lfc = pd.melt(prism_lfc, id_vars=['DrugID', 'Drug'], var_name='DepMapID', value_name='Y').dropna(subset=['Y'])
    prism_lfc = prism_lfc.dropna()
    
    broad_to_name = pd.Series(prism_lfc['Drug'].values, index=prism_lfc['DrugID']).to_dict()
    
    # everything that has the same DrugID and same DepMapID should be collapsed into one row
    prism_lfc = prism_lfc[['DrugID', 'DepMapID', 'Y']].groupby(['DrugID', 'DepMapID']).mean().reset_index()
    prism_lfc['DrugName'] = prism_lfc['DrugID'].apply(lambda x: broad_to_name[x])
    name_to_broad = pd.Series(prism_lfc['DrugID'].values, index=prism_lfc['DrugName']).to_dict()
    
    # collapse same DrugName and DepMapID
    prism_lfc = prism_lfc[['DrugName', 'DepMapID', 'Y']].groupby(['DrugName', 'DepMapID']).mean().reset_index()
    prism_lfc['DrugID'] = prism_lfc['DrugName'].apply(lambda x: name_to_broad[x])
    
    prism_lfc = prism_lfc.rename(columns={'DrugName': 'Drug', 'DrugID': 'BroadID'})

    # create a numericID for each DrugID
    drug_to_id = pd.Series(range(len(prism_lfc['BroadID'].unique())), index=prism_lfc['BroadID'].unique()).to_dict()
    prism_lfc['DrugID'] = prism_lfc['BroadID'].apply(lambda x: drug_to_id[x])
    
    prism_lfc = pd.merge(prism_lfc, model, on='DepMapID', how='inner')
    
    return prism_lfc


def obtain_metadata(dataset='gdsc', path='./data', drug_threshold=10, **kwargs):
    """
    Load metadata for the specified dataset.
    
    Args:
        dataset: 'gdsc' or 'prism'
        path: Path to the data directory
        drug_threshold: Minimum number of cell lines per drug
        
    Returns:
        DataFrame with metadata
    """
    if dataset == 'gdsc':
        return obtain_gdsc(path, drug_threshold, **kwargs)
    
    if dataset == 'prism':
        return obtain_prism_lfc(path, drug_threshold, **kwargs)


def obtain_drugs_metadata(dataset='gdsc', path='./data'):
    """
    Get drug metadata including MOA and targets.
    
    Args:
        dataset: 'gdsc' or 'prism'
        path: Path to the data directory
        
    Returns:
        DataFrame with drug metadata
    """
    if dataset == 'gdsc':
        return get_gdsc_drugs_metadata(path)
    
    if dataset == 'prism':
        return get_prism_lfc_drugs_metadata(path)


def get_gdsc_drugs_metadata(data_path='./data'):
    """
    Get GDSC drug metadata.
    
    Args:
        data_path: Path to the data directory
        
    Returns:
        DataFrame with columns: DrugID, Drug, MOA, repurposing_target
    """
    data_path = Path(data_path)
    data = pd.read_csv(data_path / 'metadata' / 'GDSC_drugs.csv')
    data = data.rename(columns={
        'DRUG_ID': 'DrugID',
        'DRUG_NAME': 'Drug',
        'PATHWAY_NAME': 'MOA',
        'HGCN_TARGETS': 'repurposing_target'
    })
    return data


def get_prism_lfc_drugs_metadata(data_path='./data'):
    """
    Get PRISM drug metadata.
    
    Args:
        data_path: Path to the data directory
        
    Returns:
        DataFrame with drug metadata
    """
    data_path = Path(data_path)
    data = obtain_prism_lfc(data_path)
    drugs = data[['Drug', 'BroadID', 'DrugID']].drop_duplicates()
    prism_metadata = pd.read_csv(data_path / 'metadata' / 'Repurposing_Public_23Q2_Extended_Primary_Compound_List.csv')
    out = pd.merge(drugs, prism_metadata[['IDs', 'MOA', 'repurposing_target']], 
                   left_on='BroadID', right_on='IDs', how='left').drop(columns=['IDs'])
    out = out.drop_duplicates()
    return out


class DatasetLoader:
    """
    A class for loading and managing transcriptomic and drug sensitivity data.
    
    This class provides functionality similar to CellHit's DatasetLoader,
    allowing for data splitting, scaling, and preparation for model training.
    """

    def __init__(self, dataset='gdsc', data_path='metadata.csv',
                 celligner_output_path='celligner_CCLE_TCGA.feather',
                 use_external_datasets=False, filter_projects=None,
                 samp_x_tissue=2, random_state=0, **kwargs):
        """
        Initialize DatasetLoader.
        
        Args:
            dataset: 'gdsc' or 'prism'
            data_path: Path to metadata directory
            celligner_output_path: Path to Celligner output feather file
            use_external_datasets: Whether to use external (non-CCLE) transcriptomic data
            filter_projects: List of TCGA project IDs to keep
            samp_x_tissue: Number of samples per tissue type for validation set
            random_state: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.celligner_output_path = Path(celligner_output_path)
        self.random_state = random_state
        self.samp_x_tissue = samp_x_tissue

        # obtain metadata for the selected dataset
        self.metadata = obtain_metadata(dataset=dataset, path=self.data_path)

        # load all transcriptomic data from the Celligner output
        if filter_projects is not None:
            print(f"Filtering TCGA samples to keep only projects: {filter_projects}")
            self.all_transcriptomics_data = pd.read_feather(self.celligner_output_path)
            try:
                tcga_meta_path = Path(data_path) / 'metadata' / 'tcga_clinical.tsv'
                tcga_meta = pd.read_csv(tcga_meta_path, sep='\t')
                ids_to_keep_tcga = set(tcga_meta[tcga_meta['project_id'].isin(filter_projects)]['case_submitter_id'])
                ccle_data = self.all_transcriptomics_data[self.all_transcriptomics_data['Source'] == 'CCLE']
                tcga_data = self.all_transcriptomics_data[self.all_transcriptomics_data['Source'] == 'TCGA']
                tcga_data_filtered = tcga_data[tcga_data['index'].isin(ids_to_keep_tcga)]
                print(f"After filtering, {len(tcga_data_filtered)} TCGA samples remain.")
                self.all_transcriptomics_data = pd.concat([ccle_data, tcga_data_filtered], ignore_index=True)
            except FileNotFoundError:
                print(f"Warning: TCGA metadata file not found. Cannot filter by project.")
            except Exception as e:
                print(f"An error occurred during TCGA filtering: {e}")
            self.source_mapper = self.all_transcriptomics_data[['index', 'Source']]
            self.cell_lines_data = self.all_transcriptomics_data[self.all_transcriptomics_data['Source'] == 'CCLE'].drop(columns=['Source']).set_index('index')
            self.all_transcriptomics_data = self.all_transcriptomics_data.drop(columns=['Source']).set_index('index')
        else:
            all_transcriptomics_data = pd.read_feather(self.celligner_output_path)
            self.cell_lines_data = all_transcriptomics_data[all_transcriptomics_data['Source'] == 'CCLE'].drop(columns=['Source']).set_index('index')

        self.x_are_scaled = False
        self.genes = self.cell_lines_data.columns
        
        # consider pairs only of cell lines for which we have transcriptomic data
        self.metadata = self.metadata[self.metadata['DepMapID'].isin(set(self.cell_lines_data.index))]

    def split_and_scale(self, drugID=None, val_split=True, val_random_state=0, 
                        use_external=False, scale_full_metadata=False, pre_scaling=True):
        """
        Split data into train/validation/test sets and scale features.
        
        Args:
            drugID: Filter for specific drug (optional)
            val_split: Whether to create validation set
            val_random_state: Random seed for validation split
            use_external: Whether to use external data
            scale_full_metadata: Whether to scale all metadata
            pre_scaling: Whether to scale before splitting
            
        Returns:
            Dictionary with train_X, train_Y, test_X, test_Y, and optionally valid_X, valid_Y
        """
        # shuffle the data and take the first 2 samples for each tissue type as test set
        test_depmapIDs = set(self.metadata.sample(frac=1, random_state=self.random_state)
                             .groupby('OncotreeLineage').head(self.samp_x_tissue)
                             .reset_index()['DepMapID'].values)
        train_depmapIDs = set(self.metadata['DepMapID'].values) - set(test_depmapIDs)

        self.meta_train = self.metadata[self.metadata['DepMapID'].isin(train_depmapIDs)]
        self.meta_test = self.metadata[self.metadata['DepMapID'].isin(test_depmapIDs)]

        if pre_scaling:
            self._scale(train_depmapIDs, use_external=use_external)

        if drugID is not None:
            self.meta_train = self.meta_train[self.meta_train['DrugID'] == int(drugID)]
            self.meta_test = self.meta_test[self.meta_test['DrugID'] == int(drugID)]

        if val_split:
            valid_depmapIDs = set(self.meta_train.sample(frac=1, random_state=val_random_state)
                                  .groupby('OncotreeLineage').head(self.samp_x_tissue)
                                  .reset_index()['DepMapID'])
            train_depmapIDs = set(self.meta_train['DepMapID'].values) - set(valid_depmapIDs)
            self.meta_valid = self.meta_train[self.meta_train['DepMapID'].isin(valid_depmapIDs)]
            self.meta_train = self.meta_train[self.meta_train['DepMapID'].isin(train_depmapIDs)]
        
        if not pre_scaling:
            self._scale(train_depmapIDs, use_external=use_external)
       
        # Apply Y scaling and formatting
        self.meta_train['Y'] = self.meta_train.apply(
            lambda x: (x['Y'] - self.drug_mean_dict[x['DrugID']]) / self.drug_std_dict[x['DrugID']], axis=1)
        self.meta_test['Y'] = self.meta_test.apply(
            lambda x: (x['Y'] - self.drug_mean_dict[x['DrugID']]) / self.drug_std_dict[x['DrugID']], axis=1)
        
        if val_split:
            self.meta_valid['Y'] = self.meta_valid.apply(
                lambda x: (x['Y'] - self.drug_mean_dict[x['DrugID']]) / self.drug_std_dict[x['DrugID']], axis=1)

        if scale_full_metadata:
            self.scaled_metadata = self.metadata.copy()
            self.scaled_metadata['Y'] = self.metadata.apply(
                lambda x: (x['Y'] - self.drug_mean_dict[x['DrugID']]) / self.drug_std_dict[x['DrugID']], axis=1)

        # Prepare output
        train_X = self.Xs[list(self.meta_train['DepMapID'].values)]
        train_X = pd.DataFrame(train_X, columns=self.genes, index=self.meta_train['DepMapID'].values)

        test_X = self.Xs[list(self.meta_test['DepMapID'].values)]
        test_X = pd.DataFrame(test_X, columns=self.genes, index=self.meta_test['DepMapID'].values)

        train_Y = pd.Series(self.meta_train['Y'].values, index=self.meta_train['DrugID'].values)
        test_Y = pd.Series(self.meta_test['Y'].values, index=self.meta_test['DrugID'].values)

        out_values = {'train_X': train_X, 'train_Y': train_Y, 'test_X': test_X, 'test_Y': test_Y}

        if val_split:
            valid_X = self.Xs[list(self.meta_valid['DepMapID'].values)]
            valid_X = pd.DataFrame(valid_X, columns=self.genes, index=self.meta_valid['DepMapID'].values)
            valid_Y = pd.Series(self.meta_valid['Y'].values, index=self.meta_valid['DrugID'].values)
            out_values['valid_X'] = valid_X
            out_values['valid_Y'] = valid_Y
        
        if use_external:
            external_ids = self.Xs.get_all_keys()
            external_X = self.Xs[external_ids]
            external_X = pd.DataFrame(external_X, columns=self.genes, index=external_ids)
            out_values['external_X'] = external_X

        return out_values

    def _scale(self, train_depmapIDs, use_external=False):
        """Internal method to scale data."""
        if self.x_are_scaled:
            self._revert_scaling(use_external=use_external)

        if not use_external:
            self.cell_mean = self.cell_lines_data[self.cell_lines_data.index.isin(train_depmapIDs)].mean()
            self.cell_std = self.cell_lines_data[self.cell_lines_data.index.isin(train_depmapIDs)].std()
            self.cell_lines_data = (self.cell_lines_data - self.cell_mean) / self.cell_std
        
        if use_external:
            self.cell_mean = self.cell_lines_data.mean()
            self.cell_std = self.cell_lines_data.std()
            self.cell_lines_data = (self.cell_lines_data - self.cell_mean) / self.cell_std
            self.all_transcriptomics_data = (self.all_transcriptomics_data - self.cell_mean) / self.cell_std
            all_lines_dict = {cid: np.array(cell).reshape(1, -1) 
                              for cid, cell in zip(self.all_transcriptomics_data.index, 
                                                  self.all_transcriptomics_data.values)}
            self.Xs_pos_name_mapper = {pos: col for pos, col in enumerate(self.all_transcriptomics_data.columns)}
            self.Xs = IndexedArray(all_lines_dict)
        else:
            cell_lines_dict = {cid: np.array(cell).reshape(1, -1) 
                               for cid, cell in zip(self.cell_lines_data.index, self.cell_lines_data.values)}
            self.Xs_pos_name_mapper = {pos: col for pos, col in enumerate(self.cell_lines_data.columns)}
            self.Xs = IndexedArray(cell_lines_dict)

        self.x_are_scaled = True

        # Y scaling and formatting
        self.drug_mean_dict = self.meta_train[['DrugID', 'Y']].groupby('DrugID').mean()
        self.drug_mean_dict = pd.Series(data=self.drug_mean_dict['Y'].values, index=self.drug_mean_dict.index).to_dict()
        self.drug_std_dict = self.meta_train[['DrugID', 'Y']].groupby('DrugID').std()
        self.drug_std_dict = pd.Series(data=self.drug_std_dict['Y'].values, index=self.drug_std_dict.index).to_dict()

    def _revert_scaling(self, use_external=False):
        """Internal method to revert scaling."""
        assert self.x_are_scaled, "Data is not scaled"

        if use_external:
            self.cell_lines_data = self.cell_lines_data * self.cell_std + self.cell_mean
            self.all_transcriptomics_data = self.all_transcriptomics_data * self.cell_std + self.cell_mean
        else:
            self.cell_lines_data = self.cell_lines_data * self.cell_std + self.cell_mean

    def get_genes(self):
        """Return gene names."""
        return self.genes
    
    def get_drugs_ids(self):
        """Return unique drug IDs."""
        return self.metadata['DrugID'].unique()
    
    def get_drugs_names(self):
        """Return unique drug names."""
        return self.metadata['Drug'].unique()
    
    def get_drug_name(self, drugID):
        """Get drug name from drug ID."""
        if not hasattr(self, 'drug_name_dict'):
            mapping_data = self.metadata[['Drug', 'DrugID']].drop_duplicates()
            mapping_data['DrugID'] = mapping_data['DrugID'].astype(int)
            self.drug_name_dict = pd.Series(data=mapping_data['Drug'].values, 
                                            index=mapping_data['DrugID']).to_dict()
        return self.drug_name_dict[int(drugID)]
    
    def get_drug_id(self, drug_name):
        """Get drug ID from drug name."""
        if not hasattr(self, 'drug_id_dict'):
            mapping_data = self.metadata[['Drug', 'DrugID']].drop_duplicates()
            mapping_data['DrugID'] = mapping_data['DrugID'].astype(int)
            self.drug_id_dict = pd.Series(data=mapping_data['DrugID'].values, 
                                          index=mapping_data['Drug']).to_dict()
        return self.drug_id_dict[drug_name]
    
    def get_drug_mean(self, drugID):
        """Get drug mean IC50."""
        return self.drug_mean_dict[drugID]
    
    def get_drug_std(self, drugID):
        """Get drug std IC50."""
        return self.drug_std_dict[drugID]
    
    def get_indexes_sources(self, indexes):
        """Get source (CCLE/TCGA) for given indexes."""
        return self.source_mapper.set_index('index').loc[indexes]['Source'].values
