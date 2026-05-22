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

    # --- 2. Standardize Disease Names (与之前版本相同) ---
    # (省略这部分的代码，因为它与上一版完全相同，我们只关注新的合并逻辑)
    # ... 您之前的 standardize_disease_names_hybrid 函数的核心逻辑可以放在这里 ...
    # 为了脚本的完整性，我把它全部复制过来：
    print("\n--- 2. Standardizing Disease Names ---")
    alias_map = {
        'Head and Neck squamous cell carcinoma': 'head & neck squamous cell carcinoma',
        'Kidney renal clear cell carcinoma': 'kidney clear cell carcinoma',
        'Uterine Corpus Endometrial Carcinoma': 'uterine corpus endometrioid carcinoma',
        'Cervical squamous cell carcinoma and endocervical adenocarcinoma': 'cervical & endocervical cancer',
    }
    disease_mapping = {re.sub(r'[^a-z0-9]', '', name.lower()): name for name in available_diseases_list}
    
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
                print(f"Warning: Still could not find a match for '{disease}' for drug '{row['DrugName']}'.")
        standardized_indications.append(list(matched_diseases))

    nci_df['DrugIndicationsStandardized'] = standardized_indications

    print("\n--- 3. Merging with GDSC Drug Information based on PubChemID ---")

    # a) 清理 NCI DataFrame 中的 PubChemID
    nci_df['PubChemID_Clean'] = nci_df['PubChemId'].str.extract(r'(\d+)').astype(float).astype('Int64')
    
    # b) 清理 GDSC Mapping DataFrame 中的 PubChemID
    gdsc_map_df['PubChemID_Clean'] = gdsc_map_df['PubChemId'].astype(float).astype('Int64')
    
    # --- 关键修正：在合并前，显式重命名 gdsc_map_df 的列，以避免冲突和歧义 ---
    gdsc_to_merge = gdsc_map_df.rename(columns={
        'Drug': 'GDSC_DrugName', # 将 'Drug' 重命名为唯一的 'GDSC_DrugName'
        'PubChemId': 'GDSC_PubChemId_Raw' # 同样给原始ID一个唯一的名字
    })
    
    # c) 执行合并 (Merge)
    #    现在不再需要 'suffixes' 参数，因为我们已经手动处理了列名冲突
    merged_df = pd.merge(
        nci_df,
        gdsc_to_merge[['DrugID', 'GDSC_DrugName', 'PubChemID_Clean']],
        on='PubChemID_Clean',
        how='left'
    )

    # d) 清理合并后的结果
    final_df = merged_df.dropna(subset=['DrugID'])
    final_df['DrugID'] = final_df['DrugID'].astype(int)
    
    num_matched = len(final_df)
    num_total_nci = len(nci_df)
    print(f"Successfully matched {num_matched} out of {num_total_nci} NCI drugs to the GDSC dataset.")

    # --- 4. 保存最终输出 ---
    # 现在，我们使用新的、明确的列名来选择列
    output_columns = [
        'DrugID',                      # 来自 GDSC (明确)
        'GDSC_DrugName',                 # 来自 GDSC (明确)
        'PubChemID_Clean',             # 公共的合并键
        'PubChemId',                   # 来自 NCI (现在是唯一的)
        'DrugName',                    # 来自 NCI (现在是唯一的)
        'DrugIndicationsStandardized'
    ]
    
    final_df = final_df[output_columns].rename(columns={
        'PubChemID_Clean': 'PubChemID',
        'PubChemId': 'NCI_PubChemId_Raw',
        'DrugName': 'NCI_DrugName_Raw',
        'DrugIndicationsStandardized': 'Standardized_Indications'
    })
    
    print(f"\n--- 4. Saving Final Merged and Standardized Data ---")
    final_df.to_csv(output_csv_path, index=False)
    print(f"Final data successfully saved to: {output_csv_path}")
    print("\nFinal Data Head:")
    print(final_df.head())

    return final_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Standardize indications and merge with GDSC drug list.")
    
    parser.add_argument('--nci_input', default='/Users/hechang/Documents/chenlab/CellHit_onlyCPU/data/metadata/nci_compiled_dataset.csv', help="Path to the NCI compiled dataset CSV.")
    parser.add_argument('--gdsc_mapping', default="/Users/hechang/Documents/chenlab/CellHit_onlyCPU/data/metadata/gdsc_pubchem_mappings.csv", help="Path to your GDSC drug-PubChemID mapping file.")
    parser.add_argument('--output_csv', default='./gdsc_clinical_indications.csv', help="Path to save the final output.")
    
    args = parser.parse_args()

    # 在这里粘贴你的 available_diseases 列表
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