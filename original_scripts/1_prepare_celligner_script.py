import os
import re
import pickle
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm
from celligner import Celligner


def source_mapper(x,external=None):
        if x in set(ccle.index):
            return 'CCLE'
        elif x in set(tcga.index):
            return 'TCGA'
        else:
            return external


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--use_external', action='store_true', help='Use external datasets')
    parser.add_argument('--external_dataset', type=str, help='Name of the external dataset')

    args = parser.parse_args()

    ###--CCLE--###
    data_path = Path('../../data/transcriptomics')

    #Read the CCLE data
    ccle = pd.read_csv(data_path/'OmicsExpressionProteinCodingGenesTPMLogp1.csv',index_col=0)
    ccle.columns = [str(i).split(' ')[0] for i in ccle.columns]

    #compute the std of the data
    ccle_stds = ccle.apply(np.std,axis=0)
    #identify the genes with no variance and remove them
    ccle_stds = ccle_stds[ccle_stds>0]
    ccle_stds = set(ccle_stds.index)
    ccle = ccle[[i for i in ccle.columns if i in ccle_stds]]

    ###--TCGA--###
    tcga = pd.read_csv(data_path/'TumorCompendium_v11_PolyA_hugo_log2tpm_58581genes_2020-04-09.tsv',sep='\t')
    tcga = tcga.set_index('Gene').transpose()

    #remove 0 std columns
    tcga_stds = tcga.apply(np.std,axis=0)
    tcga_stds = tcga_stds[tcga_stds>0]
    tcga_stds = set(tcga_stds.index)
    tcga = tcga[[i for i in tcga.columns if i in tcga_stds]]


    common_columns = set(ccle.columns).intersection(tcga.columns)
    ccle = ccle[list(common_columns)]
    tcga = tcga[list(common_columns)]

    tumor_samples = tcga

    #align the data
    my_alligner = Celligner()
    my_alligner.fit(ccle)
    my_alligner.transform(tumor_samples)

    #save the data
    output = my_alligner.combined_output.copy()
    output['Source'] = output.index.map(lambda x: source_mapper(x,external=args.external_dataset))
    output = output[list(output.columns[0:1]) + list(output.columns[-1:]) + list(output.columns[:-1])]

    if args.use_external:
        suffix = 'CCLE_TCGA_' + args.external_dataset
    else:
        suffix = 'CCLE_TCGA'

    #save feather file and base alligner
    output.to_feather(data_path/'celligner_{suffix}.feather')
    my_alligner.save(data_path/'base_alligner_{suffix}.pkl')
