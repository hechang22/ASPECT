"""
Script 1: Prepare Celligner Alignment

This script aligns CCLE and TCGA transcriptomic data using Celligner.
It can optionally include external datasets.

Usage:
    python 1_prepare_celligner.py --use_external --external_dataset <DATASET_NAME>
    python 1_prepare_celligner.py  # Process only CCLE and TCGA
"""

import os
import re
import pickle
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from celligner import Celligner


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Align CCLE and TCGA transcriptomic data using Celligner.'
    )
    parser.add_argument(
        '--use_external', 
        action='store_true', 
        help='Use external datasets'
    )
    parser.add_argument(
        '--external_dataset', 
        type=str, 
        help='Name of the external dataset'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='../../data/transcriptomics',
        help='Path to transcriptomic data directory'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='../../data/transcriptomics',
        help='Path to save output files'
    )

    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    ###--CCLE--###
    print("Loading CCLE data...")
    ccle = pd.read_csv(
        data_path / 'OmicsExpressionProteinCodingGenesTPMLogp1.csv',
        index_col=0
    )
    ccle.columns = [str(i).split(' ')[0] for i in ccle.columns]

    # Compute std and remove genes with no variance
    ccle_stds = ccle.apply(np.std, axis=0)
    ccle_stds = ccle_stds[ccle_stds > 0]
    ccle_stds = set(ccle_stds.index)
    ccle = ccle[[i for i in ccle.columns if i in ccle_stds]]
    print(f"CCLE data loaded: {ccle.shape}")

    ###--TCGA--###
    print("Loading TCGA data...")
    tcga = pd.read_csv(
        data_path / 'TumorCompendium_v11_PolyA_hugo_log2tpm_58581genes_2020-04-09.tsv',
        sep='\t'
    )
    tcga = tcga.set_index('Gene').transpose()

    # Remove 0 std columns
    tcga_stds = tcga.apply(np.std, axis=0)
    tcga_stds = tcga_stds[tcga_stds > 0]
    tcga_stds = set(tcga_stds.index)
    tcga = tcga[[i for i in tcga.columns if i in tcga_stds]]
    print(f"TCGA data loaded: {tcga.shape}")

    # Find common columns
    common_columns = set(ccle.columns).intersection(tcga.columns)
    ccle = ccle[list(common_columns)]
    tcga = tcga[list(common_columns)]
    print(f"Common genes: {len(common_columns)}")

    tumor_samples = tcga

    # Align data using Celligner
    print("Running Celligner alignment...")
    my_alligner = Celligner()
    my_alligner.fit(ccle)
    my_alligner.transform(tumor_samples)

    # Save the data
    output = my_alligner.combined_output.copy()
    
    # Map sample IDs to their source
    ccle_ids = set(ccle.index)
    tcga_ids = set(tcga.index)
    ext_label = args.external_dataset
    def _source_mapper(x):
        if x in ccle_ids:
            return 'CCLE'
        elif x in tcga_ids:
            return 'TCGA'
        else:
            return ext_label
    
    output['Source'] = output.index.map(_source_mapper)
    output = output[list(output.columns[0:1]) + list(output.columns[-1:]) + list(output.columns[1:-1])]

    if args.use_external:
        suffix = 'CCLE_TCGA_' + args.external_dataset
    else:
        suffix = 'CCLE_TCGA'

    # Save feather file and base aligner
    output_file = output_path / f'celligner_{suffix}.feather'
    aligner_file = output_path / f'base_alligner_{suffix}.pkl'
    
    output.to_feather(output_file)
    my_alligner.save(aligner_file)
    
    print(f"Output saved to: {output_file}")
    print(f"Aligner saved to: {aligner_file}")
