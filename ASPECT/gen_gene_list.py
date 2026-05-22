"""
Gene getter for mechanism of action (MOA) based gene selection.

This module provides the GeneGetter class for retrieving drug-associated genes
from various sources including LLM predictions, ligand targets, and pathway annotations.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class GeneGetter:
    """
    A class for retrieving genes associated with drug mechanisms of action.
    
    This class aggregates genes from multiple sources:
    - LLM-predicted genes
    - Ligand-associated genes
    - Target-associated genes
    - KEGG pathway genes
    - Important/downstream pathway genes
    
    Supports both ASPECT's 'knowledge_data' folder structure and
    CellHit's 'MOA_data' folder structure.
    """

    def __init__(self, dataset='gdsc', data_path=None, available_genes=None, **kwargs):
        """
        Initialize GeneGetter.
        
        Args:
            dataset: 'gdsc' or 'prism'
            data_path: Path to the data directory
            available_genes: Set of gene names available in the expression matrix
            **kwargs: Additional arguments including 'moa_data_path' for backward compatibility
        """
        self.dataset = dataset
        self.available_genes = set(available_genes) if available_genes is not None else set()

        data_path = Path(data_path) if data_path else Path('.')
        
        # Support both folder naming conventions
        # ASPECT uses 'knowledge_data'
        # CellHit uses 'MOA_data'
        moa_data_path = kwargs.get('moa_data_path', None)
        if moa_data_path:
            # Backward compatibility with CellHit
            self.data_path = Path(moa_data_path)
        else:
            # Try ASPECT naming first, then CellHit naming
            knowledge_path = data_path / 'knowledge_data'
            moa_path = data_path / 'MOA_data'
            
            if knowledge_path.exists():
                self.data_path = knowledge_path
            elif moa_path.exists():
                self.data_path = moa_path
            else:
                # Default to knowledge_data (ASPECT convention)
                self.data_path = knowledge_path

        # Load common genes (for GDSC dataset)
        if dataset == 'gdsc':
            common_genes_path = self.data_path / 'gdsc_most_common_genes.txt'
            if common_genes_path.exists():
                with open(common_genes_path, 'r') as f:
                    self.common_genes = f.read().splitlines()
            else:
                self.common_genes = []

        # Load LLM associated genes
        llm_path = self.data_path / f'{dataset}_LLM_drugID_to_genes.json'
        if llm_path.exists():
            with open(llm_path, 'r') as f:
                self.llm_genes = json.load(f)
        else:
            self.llm_genes = {}

        # Load Ligand associated genes
        ligand_path = self.data_path / f'{dataset}_ligand_drugID_to_genes.json'
        if ligand_path.exists():
            with open(ligand_path, 'r') as f:
                self.ligand_genes = json.load(f)
        else:
            self.ligand_genes = {}

        # Load Target associated genes
        target_path = self.data_path / f'{dataset}_target_drugID_to_genes.json'
        if target_path.exists():
            with open(target_path, 'r') as f:
                self.target_genes = json.load(f)
        else:
            self.target_genes = {}

        # Load KEGG pathway genes
        kegg_path = self.data_path / f'{dataset}_KEGG.json'
        if kegg_path.exists():
            with open(kegg_path, 'r') as f:
                self.kegg_genes = json.load(f)
        else:
            self.kegg_genes = {}

        # Load downstream/important pathway genes
        ig_path = self.data_path / f'{dataset}_important_genes.json'
        if ig_path.exists():
            with open(ig_path, 'r') as f:
                self.ig_genes = json.load(f)
        else:
            self.ig_genes = {}

    def get_genes(self, drugID):
        """
        Get all genes associated with a specific drug.
        
        Args:
            drugID: Drug ID (numeric or string)
            
        Returns:
            List of gene names that are both associated with the drug
            and present in available_genes
        """
        genes = []

        # Add LLM-predicted genes
        if str(drugID) in self.llm_genes:
            genes.extend(self.llm_genes[str(drugID)])

        # Add common genes for GDSC if no LLM genes available
        if (self.dataset == 'gdsc') and (str(drugID) not in self.llm_genes):
            genes.extend(self.common_genes)

        # Add ligand genes
        if str(drugID) in self.ligand_genes:
            genes.extend(self.ligand_genes[str(drugID)])

        # Add target genes
        if str(drugID) in self.target_genes:
            genes.extend(self.target_genes[str(drugID)])

        # Add KEGG pathway genes
        if str(drugID) in self.kegg_genes:
            genes.extend(self.kegg_genes[str(drugID)])

        # Add downstream pathway genes
        if str(drugID) in self.ig_genes:
            genes.extend(self.ig_genes[str(drugID)])

        # Filter by available genes
        return list(set(genes).intersection(self.available_genes))
    
    def get_gene_sources(self, drugID):
        """
        Get genes grouped by their source.
        
        Args:
            drugID: Drug ID
            
        Returns:
            Dictionary mapping source names to lists of genes
        """
        sources = {}
        
        if str(drugID) in self.llm_genes:
            sources['llm'] = self.llm_genes[str(drugID)]
        
        if str(drugID) in self.ligand_genes:
            sources['ligand'] = self.ligand_genes[str(drugID)]
            
        if str(drugID) in self.target_genes:
            sources['target'] = self.target_genes[str(drugID)]
            
        if str(drugID) in self.kegg_genes:
            sources['kegg'] = self.kegg_genes[str(drugID)]
            
        if str(drugID) in self.ig_genes:
            sources['important'] = self.ig_genes[str(drugID)]
        
        return sources
    
    def get_all_drug_ids(self):
        """
        Get all drug IDs that have associated genes.
        
        Returns:
            List of unique drug IDs
        """
        all_ids = set()
        all_ids.update(self.llm_genes.keys())
        all_ids.update(self.ligand_genes.keys())
        all_ids.update(self.target_genes.keys())
        all_ids.update(self.kegg_genes.keys())
        all_ids.update(self.ig_genes.keys())
        return list(all_ids)
