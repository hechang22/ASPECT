"""
ASPECT: A Semantic Prediction Engine for Cancer Therapeutics

A framework for predicting drug sensitivity using transcriptomic data
and mechanism-aware gene selection.
"""

from .dataset_loaders import (
    obtain_metadata,
    obtain_gdsc,
    obtain_prism_lfc,
    obtain_drugs_metadata,
    get_gdsc_drugs_metadata,
    get_prism_lfc_drugs_metadata,
    DatasetLoader,
    IndexedArray
)

from .gen_gene_list import GeneGetter

__all__ = [
    'obtain_metadata',
    'obtain_gdsc',
    'obtain_prism_lfc',
    'obtain_drugs_metadata',
    'get_gdsc_drugs_metadata',
    'get_prism_lfc_drugs_metadata',
    'DatasetLoader',
    'IndexedArray',
    'GeneGetter'
]
