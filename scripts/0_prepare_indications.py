"""
Script 0: Prepare and Standardize Clinical Indications

This script standardizes disease names from NCI and merges with GDSC drug information.
It creates a mapping of drugs to their FDA-approved indications.

Usage:
    python 0_prepare_indications.py \
        --nci_input ./data/metadata/nci_compiled_dataset.csv \
        --gdsc_mapping ./data/metadata/gdsc_pubchem_mappings.csv \
        --output_csv ./results/gdsc_clinical_indications.csv
"""

import pandas as pd
import argparse
import ast
import re


def standardize_and_merge_indications(
    nci_indications_path,
    gdsc_mapping_path,
    available_diseases_list,
    output_csv_path
):
    """
    Standardizes disease names and merges GDSC drug information based on PubChemID.
    """
    print("--- 1. Loading Input Files ---")
    try:
        nci_df = pd.read_csv(nci_indications_path)
        gdsc_map_df = pd.read_csv(gdsc_mapping_path)
        print("Successfully loaded NCI indications and GDSC mapping files.")
    except FileNotFoundError as e:
        print(f"Error: A required file was not found. {e}")
        return

    # Disease name aliases for standardization
    print("\n--- 2. Standardizing Disease Names ---")
    alias_map = {
        'Head and Neck squamous cell carcinoma': 'head & neck squamous cell carcinoma',
        'Kidney renal clear cell carcinoma': 'kidney clear cell carcinoma',
        'Uterine Corpus Endometrial Carcinoma': 'uterine corpus endometrioid carcinoma',
        'Cervical squamous cell carcinoma and endocervical adenocarcinoma': 'cervical & endocervical cancer',
    }
    
    disease_mapping = {
        re.sub(r'[^a-z0-9]', '', name.lower()): name 
        for name in available_diseases_list
    }
    
    standardized_indications = []
    for index, row in nci_df.iterrows():
        try:
            extended_indications_str = row['DrugIndicationsExtended']
            if pd.isna(extended_indications_str) or extended_indications_str in ["None found", "N/A"]:
                standardized_indications.append([])
                continue
            original_diseases = ast.literal_eval(extended_indications_str) if extended_indications_str.startswith('[') else [extended_indications_str]
        except Exception:
            original_diseases = [d.strip() for d in str(row['DrugIndicationsExtended']).split(',')]
        
        matched_diseases = set()
        for disease in original_diseases:
            corrected_disease = alias_map.get(disease, disease)
            search_key = re.sub(r'[^a-z0-9]', '', corrected_disease.lower())
            found = False
            if search_key in disease_mapping:
                matched_diseases.add(disease_mapping[search_key])
                found = True
            if not found:
                for key, standard_name in disease_mapping.items():
                    if search_key in key or key in search_key:
                        matched_diseases.add(standard_name)
                        found = True
                        break
            if not found:
                print(f"Warning: Could not find match for '{disease}' for drug '{row['DrugName']}'.")
        
        standardized_indications.append(list(matched_diseases))

    nci_df['DrugIndicationsStandardized'] = standardized_indications

    print("\n--- 3. Merging with GDSC Drug Information ---")

    # Clean PubChemIDs
    nci_df['PubChemID_Clean'] = nci_df['PubChemId'].str.extract(r'(\d+)').astype(float).astype('Int64')
    gdsc_map_df['PubChemID_Clean'] = gdsc_map_df['PubChemId'].astype(float).astype('Int64')
    
    # Rename columns to avoid conflicts
    gdsc_to_merge = gdsc_map_df.rename(columns={
        'Drug': 'GDSC_DrugName',
        'PubChemId': 'GDSC_PubChemId_Raw'
    })
    
    # Merge
    merged_df = pd.merge(
        nci_df,
        gdsc_to_merge[['DrugID', 'GDSC_DrugName', 'PubChemID_Clean']],
        on='PubChemID_Clean',
        how='left'
    )

    # Clean results
    final_df = merged_df.dropna(subset=['DrugID'])
    final_df['DrugID'] = final_df['DrugID'].astype(int)
    
    num_matched = len(final_df)
    num_total_nci = len(nci_df)
    print(f"Successfully matched {num_matched} out of {num_total_nci} NCI drugs to GDSC.")

    # Select and rename output columns
    output_columns = [
        'DrugID',
        'GDSC_DrugName',
        'PubChemID_Clean',
        'PubChemId',
        'DrugName',
        'DrugIndicationsStandardized'
    ]
    
    final_df = final_df[output_columns].rename(columns={
        'PubChemID_Clean': 'PubChemID',
        'PubChemId': 'NCI_PubChemId_Raw',
        'DrugName': 'NCI_DrugName_Raw',
        'DrugIndicationsStandardized': 'Standardized_Indications'
    })
    
    print(f"\n--- 4. Saving Results ---")
    final_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to: {output_csv_path}")
    print("\nPreview:")
    print(final_df.head())

    return final_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Standardize indications and merge with GDSC drug list."
    )
    
    parser.add_argument(
        '--nci_input',
        type=str,
        default='./data/metadata/nci_compiled_dataset.csv',
        help="Path to NCI compiled dataset CSV"
    )
    parser.add_argument(
        '--gdsc_mapping',
        type=str,
        default="./data/metadata/gdsc_pubchem_mappings.csv",
        help="Path to GDSC drug-PubChemID mapping file"
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='./results/gdsc_clinical_indications.csv',
        help="Path to save output"
    )
    
    args = parser.parse_args()

    # Available diseases list (TCGA cancer types)
    available_diseases = [
        'Ewing sarcoma',
        'INI-deficient soft tissue sarcoma NOS',
        'NUT midline carcinoma',
        'PEComa',
        'Sertoli-Leydig cell tumor, retiform',
        'acinar cell carcinoma',
        'acute leukemia',
        'acute leukemia of ambiguous lineage',
        'acute lymphoblastic leukemia',
        'acute megakaryoblastic leukemia',
        'acute myeloid leukemia',
        'acute undifferentiated leukemia',
        'adrenocortical adenoma',
        'adrenocortical cancer',
        'adrenocortical carcinoma',
        'alveolar rhabdomyosarcoma',
        'alveolar soft part sarcoma',
        'angiosarcoma',
        'atypical teratoid/rhabdoid tumor',
        'bladder urothelial carcinoma',
        'breast invasive carcinoma',
        'cervical & endocervical cancer',
        'cholangiocarcinoma',
        'choroid plexus carcinoma',
        'chronic myelogenous leukemia (S02), acute lymphoblastic leukemia (S01)',
        'clear cell carcinoma of cervix',
        'colon adenocarcinoma',
        'craniopharyngioma',
        'dedifferentiated liposarcoma',
        'desmoplastic small round cell tumor',
        'diffuse large B-cell lymphoma',
        'dysembryoplastic neuroepithelial tumor',
        'embryonal rhabdomyosarcoma',
        'embryonal tumor with multilayered rosettes',
        'endometrial stromal sarcoma',
        'ependymoma',
        'epithelioid hemangioendothelioma',
        'epithelioid sarcoma',
        'esophageal carcinoma',
        'fibrolamellar hepatocellular carcinoma',
        'fibromatosis',
        'follicular neoplasm',
        'ganglioglioma',
        'gastrointestinal stromal tumor',
        'germ cell tumor',
        'glioblastoma multiforme',
        'glioma',
        'gliomatosis cerebri',
        'head & neck squamous cell carcinoma',
        'hepatoblastoma',
        'hepatocellular carcinoma',
        'infantile fibrosarcoma',
        'inflammatory myofibroblastic tumor',
        'juvenile myelomonocytic leukemia',
        'kidney chromophobe',
        'kidney clear cell carcinoma',
        'kidney papillary cell carcinoma',
        'leiomyosarcoma',
        'leukemia',
        'lipoblastomatosis',
        'lung adenocarcinoma',
        'lung squamous cell carcinoma',
        'lymphoma',
        'malignant peripheral nerve sheath tumor',
        'medulloblastoma',
        'melanoma',
        'melanotic neuroectodermal tumor',
        'meningioma',
        'mesothelioma',
        'myeloid neoplasm NOS',
        'myeloproliferative neoplasm',
        'myoepithelial carcinoma',
        'myofibromatosis',
        'myxofibrosarcoma',
        'nasopharyngeal carcinoma',
        'neoplasm (uncertain whether benign or malignant)',
        'neuroblastoma',
        'neuroendocrine carcinoma',
        'neurofibroma',
        'neurofibromatosis type 1',
        'osteosarcoma',
        'ovarian serous cystadenocarcinoma',
        'pancreatic adenocarcinoma',
        'pheochromocytoma & paraganglioma',
        'pineal parenchymal tumor',
        'pleomorphic myxoid liposarcoma',
        'pleuropulmonary blastoma',
        'prostate adenocarcinoma',
        'rectum adenocarcinoma',
        'retinoblastoma',
        'rhabdoid tumor',
        'rhabdomyosarcoma',
        'rosette forming glioneuronal tumor',
        'sarcoma',
        'sclerosing epithelioid fibrosarcoma',
        'skin cutaneous melanoma',
        'spindle cell/sclerosing rhabdomyosarcoma',
        'stomach adenocarcinoma',
        'supratentorial embryonal tumor NOS',
        'synovial sarcoma',
        'teratoma',
        'testicular germ cell tumor',
        'thymic carcinoma',
        'thymoma',
        'thyroid carcinoma',
        'undifferentiated hepatic sarcoma',
        'undifferentiated pleomorphic sarcoma',
        'undifferentiated sarcoma NOS',
        'undifferentiated spindle cell sarcoma',
        'uterine carcinosarcoma',
        'uterine corpus endometrioid carcinoma',
        'uveal melanoma',
        'wilms tumor'
    ]
    
    standardize_and_merge_indications(
        args.nci_input,
        args.gdsc_mapping,
        available_diseases,
        args.output_csv
    )
