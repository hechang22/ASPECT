# =============================================================================
# ASPECT Downstream Analysis (Data Processing)
# =============================================================================
# This script runs all analysis computations and saves the workspace.
# Run this BEFORE 6_figures.qmd to generate all data.
# =============================================================================

library(tidyverse) 
library(janitor)
library(dplyr)
library(tidyr)
library(tibble)
library(ggplot2)
library(ggpubr)
library(TCGAbiolinks) 
library(patchwork)
library(mlr3)
library(mlr3learners)
library(mlr3viz)
library(ranger) # Efficient random forest implementation
library(survival)
library(survminer)
library(forestmodel) # For drawing forest plots
library(broom)
library(psych) # For factor analysis
library(stringr)



file_list <- list(
  "CellHit" = "/Users/hechang/Documents/chenlab/CellHit_onlyCPU/scripts/feat_C2s/validation_per_drug/all_drugs_validation_summary.csv",
  "2k+2k" = "/Users/hechang/Documents/chenlab/CellHit_onlyCPU/scripts/feat_C2s/validation_per_drug_2k_reversed/all_drugs_validation_summary_singlefile.csv",
  "Comb" = "/Users/hechang/Documents/chenlab/CellHit_onlyCPU/scripts/feat_C2s/validation_per_drug_cclev2_reversed/all_drugs_validation_summary_singlefile.csv"
)

# -------------------------------------------------------------------------
# Step 1: Read and merge all CSV files
# -------------------------------------------------------------------------
# We use purrr::map_dfr to iterate over each file in the list,
# and merge them into one data frame.
# .id = "model_name" automatically creates a column named "model_name",
# populated with the names from file_list (e.g., "Model_A", "Model_B").

all_data <- map_dfr(file_list, ~ read_csv(.x) %>% clean_names(), .id = "model_name")

# clean_names() automatically converts "Recall @ Top-N" to "recall_at_top_n",
# and "AUC" to "auc", making references cleaner.

# Inspect the merged data
print(all_data)

# -------------------------------------------------------------------------
# Step 4: (optional but recommended) Filter drugs of interest
# -------------------------------------------------------------------------
# If your file has hundreds of drugs, plotting all at once is messy.
# Typically you'd filter to a few specific drugs for comparison.

# Select drugs to compare from the example data
#drugs_to_compare <- c("Vinblastine","Irinotecan","Sorafenib","Niraparib")
drugs_to_compare = all_data$drug_name %>%unique()
  
plot_data <- all_data %>%
  filter(drug_name %in% drugs_to_compare) %>% 
  mutate(model_name= factor(model_name, levels = c('CellHit','2k+2k','Comb')))

# *** Choose the model you want to use as the sorting baseline ***
REFERENCE_MODEL <- "CellHit" # You can change this to "Model_B" or any name in file_list

# --- Sort for AUC chart ---

# 1. Filter data for the reference model
# 2. Sort by `auc` descending
# 3. Pull the `drug_name` column as a vector. This is the desired order.
drug_order_auc <- plot_data %>%
  filter(model_name == REFERENCE_MODEL) %>%
  arrange(desc(auc)) %>%
  pull(drug_name)


# 4. Use this order to mutate drug_name in plot_data
# Convert it to a factor with explicit levels
plot_data_sorted_auc <- plot_data %>%
  mutate(drug_name = factor(drug_name, levels = drug_order_auc))


# --- (optional) Create a *different* sort for the Recall chart ---
# Note: Recall ranking may differ from AUC, so we compute separately
drug_order_recall <- plot_data %>%
  filter(model_name == REFERENCE_MODEL) %>%
  arrange(desc(recall_top_n)) %>%
  pull(drug_name)

plot_data_sorted_recall <- plot_data %>%
  mutate(drug_name = factor(drug_name, levels = drug_order_recall))

# -------------------------------------------------------------------------
# Step 5: Draw grouped bar chart (compare AUC)
# -------------------------------------------------------------------------
# [Plot code removed -> see figures.qmd]

# Display the chart
# [Plot code removed -> see figures.qmd]
# -------------------------------------------------------------------------
# Step 6: Draw grouped bar chart (compare Recall)
# -------------------------------------------------------------------------
# Almost identical code; just change `y = auc` to `y = recall_at_top_n`

# [Plot code removed -> see figures.qmd]

# Display the chart
# [Plot code removed -> see figures.qmd]

######
pred_full = read_delim("/Users/hechang/Documents/chenlab/CellHit_onlyCPU/scripts/feat_C2s/predictions_comb.csv")
ground_truth = read_delim("/Users/hechang/Documents/chenlab/CellHit_onlyCPU/scripts/feat_C2s/validation_per_drug_cclev2/ground_truth_sensitivity_matrix.csv")
ground_truth = ground_truth %>% pivot_longer(-DrugName,names_to = 'SampleID', values_to = "Truth")
pred_full = pred_full %>% left_join(ground_truth)

ccle_meta = read_delim("/Users/hechang/Documents/chenlab/CellHit_onlyCPU/data/metadata/Model.csv")
tcga_md = read_delim("/Users/hechang/Documents/chenlab/CellHit_onlyCPU/data/metadata/clinical_TumorCompendium_v11_PolyA_for_GEO_20240520.tsv")

est_score = read_delim("/Users/hechang/Documents/chenlab/data/TCGA_estimate_score.gct")

DRUG = 'Vinblastine'

pred_top600 <- pred_full %>%
  dplyr::filter(DrugName == DRUG) %>%
  dplyr::arrange(-Predicted_IC50) %>%
  dplyr::slice_head(n = 600)

sub_neighbors <- pred_top600 %>%
  dplyr::filter(Truth == 1) %>%
  tidyr::separate_rows(KNN_Neighbors, sep = ";") %>%
  dplyr::filter(KNN_Neighbors != "" & !is.na(KNN_Neighbors)) %>%
  dplyr::count(KNN_Neighbors, sort = TRUE, name = "Count")

##### meta
neighbor_annotated <- sub_neighbors %>%
  left_join(ccle_meta, by = c("KNN_Neighbors" = "ModelID"))

analysis_population <- neighbor_annotated %>%
  uncount(Count) 

background_population <- ccle_meta

obs_counts <- analysis_population %>%
  filter(!is.na(PrimaryOrMetastasis)) %>%
  count(PrimaryOrMetastasis) %>%
  mutate(Group = "Selected_Neighbors")

# Extract background distribution
bg_counts <- background_population %>%
  filter(!is.na(PrimaryOrMetastasis)) %>%
  count(PrimaryOrMetastasis) %>%
  mutate(Group = "Background_CCLE")

# Merge
contingency_data <- bind_rows(obs_counts, bg_counts) %>%
  pivot_wider(names_from = PrimaryOrMetastasis, values_from = n, values_fill = 0) %>%
  column_to_rownames("Group")

# 2. Execute chi-square test
chisq_result <- chisq.test(contingency_data)

# 3. Output results
print(chisq_result)
print(chisq_result$observed) # Observed values
print(chisq_result$expected) # Expected values

plot_data <- bind_rows(obs_counts, bg_counts) %>%
  group_by(Group) %>%
  mutate(Prop = n / sum(n)) %>%
  rename(Status = PrimaryOrMetastasis)

# [Plot code removed -> see figures.qmd]


##### lineage

lineage_stat <- analysis_population %>%
  count(OncotreeLineage, sort = TRUE) %>%
  mutate(Freq_Selected = n / sum(n))

bg_stat <- background_population %>%
  count(OncotreeLineage) %>%
  mutate(Freq_BG = n / sum(n))

enrichment <- lineage_stat %>%
  left_join(bg_stat, by = "OncotreeLineage", suffix = c("_Obs", "_Bg")) %>%
  mutate(Fold_Change = Freq_Selected / Freq_BG) %>%
  arrange(desc(Fold_Change))

print(enrichment)

full_data <- pred_full %>%
  dplyr::filter(Truth == 1) %>%
  tidyr::separate_rows(KNN_Neighbors, sep = ";") %>%
  dplyr::filter(KNN_Neighbors != "" & !is.na(KNN_Neighbors)) %>%
  left_join(tcga_md, by = c("SampleID"="th_dataset_id")) %>%
  left_join(ccle_meta, by = c('KNN_Neighbors'='ModelID'))

bg_stats <- ccle_meta %>%
  count(OncotreeLineage) %>%
  mutate(Freq_BG = n / sum(n)) %>%
  select(OncotreeLineage, Freq_BG)

target_cancers <- table(tcga_md$disease) %>% sort(decreasing=T) %>% head(15) %>% names()

plot_data_list <- list()

for (cancer in target_cancers) {
  
  # 3.1 Extract all neighbors for this cancer type
  cancer_neighbors <- full_data %>%
    filter(disease == cancer)
  
  total_neighbors <- nrow(cancer_neighbors)
  
  # 3.2 Compute lineage frequency of selected neighbors
  obs_stats <- cancer_neighbors %>%
    count(OncotreeLineage) %>%
    mutate(Freq_Selected = n / sum(n)) %>%
    rename(n_Obs = n)
  
  # 3.3 Merge background and compute Fold Change
  enrichment_res <- obs_stats %>%
    left_join(bg_stats, by = "OncotreeLineage") %>%
    mutate(
      Fold_Change = Freq_Selected / Freq_BG,
      CancerType = cancer # label this cancer type column
    ) %>%
    # Keep only major enrichments (FC > 1) or highlight lineages
    filter(n_Obs > 0) # filter out noise
  
  plot_data_list[[cancer]] <- enrichment_res
}

# Merge all results
final_plot_data <- bind_rows(plot_data_list)

highlight_lineages <- c("Esophagus/Stomach", "Breast", "Pancreas", "Ovary/Fallopian Tube")

viz_data <- final_plot_data %>%
  # 1. Show only high Fold Change or highlight lineages
  filter(OncotreeLineage %in% highlight_lineages | Fold_Change > 2.0) %>%
  
  # 2. Put the most significant (Stomach) at the top
  mutate(OncotreeLineage = fct_relevel(OncotreeLineage, highlight_lineages)) %>%
  mutate(FC_Plot = pmin(Fold_Change, 6))

# [Plot code removed -> see figures.qmd]
  geom_point(aes(size = FC_Plot, color = FC_Plot), alpha = 0.9) +
  
  # Color: warm red palette; darker = stronger enrichment
  scale_color_gradientn(
    colors = c("grey90", "#FFB6C1", "#DC143C", "#8B0000"), 
    values = c(0, 0.2, 0.6, 1),
    name = "Fold Change"
  ) +
  
  # Size settings
  scale_size_continuous(range = c(2, 8), name = "Enrichment\nMagnitude") +
  
  # Axes and theme
  labs(
    title = "Lineage confounding analysis",
    subtitle = "CCLE lineages enriched in neighbors of patients across TCGA cancers",
    x = "TCGA Cancer Type",
    y = "CCLE Neighbor Lineage"
  ) +
  
  theme_bw() +
  theme(
    # Rotate X-axis labels to prevent overlap
    axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
    # Bold Y-axis labels
    axis.text.y = element_text(face = "bold", size = 10),
    # Subtle grid line adjustment
    panel.grid.major = element_line(color = "grey95"),
    legend.position = "right"
  ) +
  
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0.5, ymax = 4.5, 
           alpha = 0, color = "blue", linetype = "dashed", size = 1)

##### estimate

est_clean <- est_score %>%
  select(-Description) %>% # Drop Description column, keep only NAME and sample columns
  pivot_longer(
    cols = -NAME, 
    names_to = "SampleID", 
    values_to = "Score"
  ) %>%
  pivot_wider(
    names_from = NAME, 
    values_from = Score
  )

cor_data <- est_clean %>%
  inner_join(pred_full%>%filter(DrugName==DRUG), by = "SampleID")

plot_data <- cor_data %>%
  mutate(Purity_Group = cut(TumorPurity, 
                            breaks = c(0, 0.6, 0.8, 1.0), 
                            labels = c("Low", "Medium", "High")))

# [Plot code removed -> see figures.qmd]
                     method = "wilcox.test") +
  scale_fill_brewer(palette = "Reds") +
  labs(
    title = "By Tumor purity",
    x = "ESTIMATE Tumor Purity Group",
    y = "Predicted Score"
  ) +
  theme_classic()

cancer_stats <- merged_data %>%
  group_by(CancerName) %>%
  # Filter cancers with too few samples (e.g., <30), otherwise correlations are unreliable
  filter(n() > 30) %>% 
  summarise(
    N = n(),
    Corr_R = cor(TumorPurity, Predicted_IC50, method = "pearson", use = "complete.obs"),
    # Use tryCatch to prevent errors in edge cases
    P_Value = tryCatch(
      cor.test(TumorPurity, Predicted_IC50, method = "pearson")$p.value,
      error = function(e) 1
    )
  ) %>%
  ungroup()

# Check the computed results
print(head(cancer_stats %>% arrange(P_Value)))

target_cancers_df <- cancer_stats %>%
  # 1. Only look at positive correlations
  filter(Corr_R > 0) %>%
  # 2. Sort by P-value ascending (most significant first)
  arrange(P_Value) %>%
  # 3. Select top 12
  slice_head(n = 12)

target_cancers <- target_cancers_df$CancerName

print("Selected Positive Correlation Cancers:")
print(target_cancers_df)

# target_cancers_df <- cancer_stats %>%
#   arrange(P_Value) %>%
#   slice_head(n = 12)
# target_cancers <- target_cancers_df$CancerName


# --- Prepare plot data ---
plot_data_subset <- merged_data %>%
  filter(CancerName %in% target_cancers) %>%
  mutate(CancerName = factor(CancerName, levels = target_cancers))

# --- Plot ---
# [Plot code removed -> see figures.qmd]
  geom_point(alpha = 0.3, color = "royalblue", size = 1) +
  
  # 2. Regression line
  geom_smooth(method = "lm", color = "red", fill = "pink", alpha = 0.5) +
  
  # 3. Statistics annotation
  stat_cor(method = "pearson", label.x.npc = "left", label.y.npc = "top", size = 3) +
  
  # 4. Facet (scales="free" is still important)
  facet_wrap(~ CancerName, scales = "free") +  
  
  labs(
    title = "Tumor Purity vs. Scores",
    x = "ESTIMATE Tumor Purity",
    y = "Predicted Score"
  ) +
  
  theme_bw() +
  theme(
    strip.background = element_rect(fill = "grey90"),
    strip.text = element_text(face = "bold", size = 8),
    axis.text = element_text(size = 8)
  )

#############################################################
subtypes_data <- PanCancerAtlas_subtypes()
head(subtypes_data)

#drugs_to_compare <- c("Vinblastine","Irinotecan","Sorafenib","Niraparib")
DRUG='Vinblastine'

clean_subtypes <- subtypes_data %>%
  select(
    SampleID = pan.samplesID,
    CancerCode = cancer.type,
    Subtype = Subtype_Selected # Or Subtype_mRNA, depending on the cancer type
  ) %>%
  # Normalize ID format: first 12-15 chars, replace . with -
  mutate(SampleID = substr(SampleID, 1, 15)) %>% 
  mutate(SampleID = gsub("\\.", "-", SampleID))

# Prepare your prediction data (Vinblastine as example)
my_pred <- pred_full %>%
  filter(DrugName == DRUG) %>% # Replace with your drug name
  mutate(SampleID = substr(SampleID, 1, 15)) %>% # Ensure ID length is consistent for matching
  mutate(SampleID = gsub("\\.", "-", SampleID))

# Merge
merged_analysis <- my_pred %>%
  inner_join(clean_subtypes, by = "SampleID")


brca_data <- merged_analysis %>%
  filter(CancerCode == "BRCA")%>%
  filter(Subtype!='BRCA.Normal')

brca_data$Subtype <- factor(brca_data$Subtype, 
                            levels = c("BRCA.LumA", "BRCA.LumB", "BRCA.Her2", "BRCA.Basal"))

# Define key comparison groups: LumA vs Basal
my_comparisons <- list(c("BRCA.LumA", "BRCA.Basal"), c("BRCA.LumA", "BRCA.LumB"))

# [Plot code removed -> see figures.qmd]
  geom_boxplot(outlier.shape = NA, alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.2, size = 1) +
  
  # 2. Color settings (red/blue palette for hot/cold tumors)
  scale_fill_brewer(palette = "RdBu", direction = -1) + # Red for Basal (Hot/Aggressive)
  
  # 3. Add statistical significance (Wilcoxon test)
  stat_compare_means(comparisons = my_comparisons, 
                     method = "wilcox.test", 
                     label = "p.signif", # Display significance stars (*, **, ***)
                     size = 5) +
  
  # 4. Add global P-value (Kruskal-Wallis)
  stat_compare_means(label.y = max(brca_data$Predicted_IC50) * 1.1) + 
  
  # 5. Title & labels
  labs(
    title = "Validation by Molecular Subtypes: BRCA",
    subtitle = "Drug: Vinblastine (Higher Score = Higher Proliferation/Sensitivity)",
    x = "PAM50 Subtype",
    y = "Predicted Score (IC50)"
  ) +
  
  theme_classic() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(face = "bold", size = 11),
    plot.title = element_text(face = "bold")
  )


ov_data <- merged_analysis %>%
  filter(grepl("OVCA", Subtype)) # Extract OV-related subtypes

# Set order: Mesenchymal vs Proliferative at opposite ends
ov_data$Subtype <- factor(ov_data$Subtype, 
                          levels = c("OVCA.Mesenchymal", "OVCA.Differentiated", 
                                     "OVCA.Immunoreactive", "OVCA.Proliferative"))

# Compare Mesenchymal (low purity) vs Proliferative (high proliferation)
my_comparisons_ov <- list(c("OVCA.Mesenchymal", "OVCA.Proliferative"))

# [Plot code removed -> see figures.qmd]
  # Color: Proliferative = Red (Hot), Mesenchymal = Blue
  scale_fill_manual(values = c("OVCA.Mesenchymal" = "#4682B4", 
                               "OVCA.Differentiated" = "grey",
                               "OVCA.Immunoreactive" = "grey",
                               "OVCA.Proliferative" = "#DC143C")) +
  
  stat_compare_means(comparisons = my_comparisons_ov, method = "wilcox.test", label = "p.signif") +
  stat_compare_means(label.y = max(ov_data$Predicted_IC50) * 1.05) +
  
  labs(
    title = "Validation in Ovarian Cancer (OV)",
    subtitle = "Drug: Vinblastine (Signal Inversion: Proliferative Subtype -> Highest IC50)",
    x = "TCGA Molecular Subtype",
    y = "Predicted Score (IC50)"
  ) +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 15, hjust = 1, face="bold"), legend.position = "none")

skcm_data <- merged_analysis %>%
  filter(grepl("SKCM", Subtype)) %>%
  filter(Subtype != "SKCM.-") # Remove uninformative entries

# Simplify subtype names for cleaner plot
skcm_data <- skcm_data %>%
  mutate(Subtype_Label = case_when(
    grepl("BRAF", Subtype) ~ "BRAF Mutant",
    grepl("RAS", Subtype) ~ "RAS Mutant",
    grepl("NF1", Subtype) ~ "NF1 Mutant",
    grepl("Triple_WT", Subtype) ~ "Triple WT",
    TRUE ~ Subtype
  ))

# Set order：WT (Low) -> BRAF (High)
skcm_data$Subtype_Label <- factor(skcm_data$Subtype_Label, 
                                  levels = c("Triple WT", "NF1 Mutant", "RAS Mutant", "BRAF Mutant"))

# [Plot code removed -> see figures.qmd]
  stat_compare_means(comparisons = list(c("Triple WT", "BRAF Mutant")), method = "wilcox.test") +
  labs(
    title = "Validation in Melanoma (SKCM)",
    subtitle = "Drug: Vinblastine (Aggressive BRAF Mutants -> Higher Predicted IC50)",
    x = "Genomic Subtype",
    y = "Predicted Score (IC50)"
  ) +
  theme_classic() +
  theme(legend.position = "none", axis.text.x = element_text(face="bold"))


prad_data <- merged_analysis %>%
  filter(grepl("PRAD", Subtype))
prad_data <- prad_data %>%
  mutate(Group = ifelse(Subtype == "PRAD.1-ERG", "ERG Fusion (+)", "Others"))

# [Plot code removed -> see figures.qmd]
  

# ==============================================================================
# 1. Define and compute CBI (unchanged)
# ==============================================================================
cytotoxic_drugs <- c(
  "Vinblastine", "Cisplatin", "Cytarabine", "Docetaxel", "Methotrexate", 
  "5-Fluorouracil", "Paclitaxel", "Irinotecan", "Oxaliplatin", 
  "Temozolomide", "Epirubicin", "Cyclophosphamide", "Mitoxantrone", 
  "Dactinomycin", "Bleomycin", "Dacarbazine", "Bleomycin (50 uM)"
)

# Compute CBI
cbi_df <- pred_full %>%
  filter(DrugName %in% cytotoxic_drugs) %>%
  group_by(DrugName) %>%
  mutate(Z_Score = scale(Predicted_IC50)) %>% 
  ungroup() %>%
  group_by(SampleID) %>%
  summarise(
    CBI = mean(Z_Score, na.rm = TRUE), 
    N_Drugs = n() 
  ) %>%
  filter(N_Drugs >= 1) %>% 
  mutate(SampleID = substr(SampleID, 1, 15)) %>%
  mutate(SampleID = gsub("\\.", "-", SampleID))

# ==============================================================================
# 2. Merge subtype data
# ==============================================================================
subtypes_data <- PanCancerAtlas_subtypes()

clean_subtypes <- subtypes_data %>%
  select(
    SampleID = pan.samplesID,
    CancerCode = cancer.type,
    Subtype = Subtype_Selected
  ) %>%
  mutate(SampleID = substr(SampleID, 1, 15)) %>% 
  mutate(SampleID = gsub("\\.", "-", SampleID))

merged_cbi <- cbi_df %>%
  inner_join(clean_subtypes, by = "SampleID")

# ==============================================================================
# 3. Plot function (generic)
# ==============================================================================
plot_cbi_validation_auto <- function(data, cancer_code, subtype_order, title_suffix) {
  
  # 1. Filter data
  plot_data <- data %>%
    filter(CancerCode == cancer_code) %>%
    filter(Subtype %in% subtype_order)
  
  # 2. Factorize and sort
  plot_data$Subtype <- factor(plot_data$Subtype, levels = subtype_order)
  
  # 3. Auto-compute significant pairs (p < 0.05)
  # Use compare_means to compute all pairwise comparisons
  stat_res <- compare_means(CBI ~ Subtype, data = plot_data, method = "wilcox.test")
  
  # Filter rows with p < 0.05
  sig_stats <- stat_res %>% filter(p < 0.05)
  
  # Build comparisons list
  if (nrow(sig_stats) > 0) {
    significant_comparisons <- lapply(1:nrow(sig_stats), function(i) {
      c(sig_stats$group1[i], sig_stats$group2[i])
    })
  } else {
    significant_comparisons <- NULL # If no significant differences, don't draw lines
  }
  
  # 4. Plot
# [Plot code removed -> see figures.qmd]
    # Global P-value (Kruskal-Wallis)
    # Position above the highest point
    stat_compare_means(label.y = max(plot_data$CBI) * 1.5, size = 4) +
    
    # Color palette
    scale_fill_viridis_d(option = "magma", begin = 0.3, end = 0.9, direction = -1) + 
    
    labs(
      title = paste0("CBI Validation: ", cancer_code),
      subtitle = paste0(title_suffix),
      x = "Molecular Subtype",
      y = "CBI Score"
    ) +
    theme_classic() +
    theme(
      legend.position = "none",
      axis.text.x = element_text(face = "bold", size = 10, angle = 20, hjust = 1),
      plot.title = element_text(face = "bold")
    )
  
  # 5. Only draw lines for significant comparisons
  if (!is.null(significant_comparisons)) {
    p <- p + stat_compare_means(
      comparisons = significant_comparisons,
      method = "wilcox.test",
      label = "p.signif",
      size = 4,
      step.increase = 0.1,  # Auto-adjust height spacing to prevent overlap
      tip.length = 0.01
    )
  }
  
  return(p)
}

# ==============================================================================
# 4. Run Validation
# ==============================================================================

# --- A. BRCA: Basal (High Proliferation) ---
# Expected: Basal highest
p1 <- plot_cbi_validation_auto(
  merged_cbi, 
  "BRCA", 
  c("BRCA.LumA", "BRCA.LumB", "BRCA.Her2", "BRCA.Basal"),
  ""
)
# [Plot code removed -> see figures.qmd]

# --- B. OV: Proliferative vs Mesenchymal ---
# Note: CancerCode is usually "OV" not "OVCA"
p2 <- plot_cbi_validation_auto(
  merged_cbi,
  "OVCA", 
  c("OVCA.Mesenchymal", "OVCA.Differentiated", "OVCA.Immunoreactive", "OVCA.Proliferative"),
  ''
)
# [Plot code removed -> see figures.qmd]

# --- C. PRAD: Subtype validation (New) ---
# PRAD subtypes are driven by gene fusions.
# PRAD.1-ERG is the most common fusion, linked to active androgen signaling.
# We compare ERG Fusion vs Others.
# Expected: ERG fusion may show higher CBI (if model captures proliferative features).

# 1. Prepare PRAD subtype order
prad_subtypes <- c("PRAD.8-other", "PRAD.5-SPOP", "PRAD.2-ETV1", "PRAD.1-ERG")

p3 <- plot_cbi_validation_auto(
  merged_cbi,
  "PRAD",
  prad_subtypes,
  ""
)
# [Plot code removed -> see figures.qmd]

# --- D. SKCM: Genotype validation (New) ---
# SKCM subtypes are based on somatic mutations.
# BRAF/RAS mutations drive sustained MAPK activation -> high proliferation.
# Triple WT typically shows slower proliferation.
# Expected: BRAF/RAS mutants show higher CBI than Triple WT.

# 1. Prepare SKCM subtype order (low to high malignancy)
skcm_subtypes <- c("SKCM.Triple_WT", "SKCM.NF1_Any_Mutants", "SKCM.RAS_Hotspot_Mutants", "SKCM.BRAF_Hotspot_Mutants")

p4 <- plot_cbi_validation_auto(
  merged_cbi,
  "SKCM",
  skcm_subtypes,
  "BRAF/RAS Mutants (High Proliferation) show higher CBI than WT"
)
# [Plot code removed -> see figures.qmd]

# --- E. Combined display ---
# [Plot code removed -> see figures.qmd]



# ==============================================================================
# 1. Data preparation
# ==============================================================================

# Assumes clean_subtypes (BRCA) and est_clean (ESTIMATE) exist.
# If est_clean missing, run the previous transpose code first.

# 1.1 Prepare BRCA subtype data
brca_subtypes <- clean_subtypes %>%
  filter(CancerCode == "BRCA") %>%
  filter(Subtype %in% c("BRCA.LumA", "BRCA.LumB", "BRCA.Her2", "BRCA.Basal")) %>%
  # Normalize IDs for merging (first 15 chars)
  mutate(SampleID_Short = substr(SampleID, 1, 15),
         Subtype = ifelse(Subtype=='BRCA.Basal','Basal','Other'))

# 1.2 Prepare ESTIMATE data
est_data_ready <- est_clean %>%
  mutate(SampleID_Short = substr(SampleID, 1, 15)) %>%
  mutate(SampleID_Short = gsub("\\.", "-", SampleID_Short))

# 1.3 Merge
brca_est_merged <- brca_subtypes %>%
  inner_join(est_data_ready, by = "SampleID_Short")

# 1.4 Set plot order (low to high malignancy)
brca_est_merged$Subtype <- factor(brca_est_merged$Subtype, 
                                  levels = c("Basal","Other"))

# ==============================================================================
# 2. Plot validation: Immune Score
# ==============================================================================
# Expected: Basal significantly higher than LumA/LumB

# [Plot code removed -> see figures.qmd]
  # Statistical test: key comparison LumA vs Basal
  stat_compare_means(comparisons = list(c("Other", "Basal")), 
                     label = "p.signif", size=5) +
  stat_compare_means(label.y = max(brca_est_merged$ImmuneScore) * 1.1) +
  
  scale_fill_brewer(palette = "RdBu", direction = -1) + # Red for Basal
  labs(
    title = "High Immune Infiltration in Basal Subtype",
    subtitle = "Higher ImmuneScore = More immune cells (Potential sensitivity signal)",
    y = "ESTIMATE Immune Score",
    x = ""
  ) +
  theme_classic() +
  theme(legend.position = "none", axis.text.x = element_blank()) # Show axis labels later when combining plots

# ==============================================================================
# 3. Validate: Tumor Purity
# ==============================================================================
# Expected: Basal purity diluted by immune cells, possibly below LumB/Her2

# [Plot code removed -> see figures.qmd]
                     label = "p.signif", size=5) +
  
  scale_fill_brewer(palette = "RdBu", direction = -1) +
  labs(
    title = "Tumor Purity Comparison",
    subtitle = "Basal purity is diluted by immune/stromal infiltration",
    y = "ESTIMATE Tumor Purity",
    x = "Molecular Subtype"
  ) +
  theme_classic() +
  theme(legend.position = "none", axis.text.x = element_text(face="bold", size=11))

# ==============================================================================
# 4. Combine charts
# ==============================================================================

# [Plot code removed -> see figures.qmd]
# combined_plot <- p_immune / p_purity

ccle_response = read_delim("/Users/hechang/Documents/chenlab/CellHit_onlyCPU/data/metadata/GDSC2_fitted_dose_response_24Jul22.csv")
ccle_res_drug = ccle_response %>%filter(DRUG_NAME=='Vinblastine')

pri = ccle_meta$SangerModelID[ccle_meta$PrimaryOrMetastasis=='Primary']
meta = ccle_meta$SangerModelID[ccle_meta$PrimaryOrMetastasis=='Metastatic']

ccle_res_drug$LN_IC50[ccle_res_drug$SANGER_MODEL_ID %in% pri]
plot_ccle_ic50_dist = data.frame(
  IC50 = ccle_res_drug$LN_IC50[ccle_res_drug$SANGER_MODEL_ID %in% pri], Status = 'Primary'
  )%>% 
  rbind(
    data.frame(
      IC50 = ccle_res_drug$LN_IC50[ccle_res_drug$SANGER_MODEL_ID %in% meta],Status='Metastatic'
      )
  )

#################
# [Plot code removed -> see figures.qmd]
  scale_fill_brewer(palette = "RdBu", direction = -1) +
  labs(
    title = "Cell line sensitivity (Vinblastine)",
    y = "lnIC50",
    x = "CCLE status"
  ) +
  theme_classic() +
  theme(legend.position = "none", axis.text.x = element_text(face="bold", size=11))

####################### classification

# Install required packages (if not already installed)
# install.packages(c("mlr3", "mlr3learners", "mlr3viz", "ranger", "ggplot2", "tidyr", "dplyr"))


# ==============================================================================
# 1. Data Preparation
# ==============================================================================

# 1.1 Prepare labels (Target: Subtype)
target_labels <- clean_subtypes %>%
  filter(CancerCode == "BRCA") %>%
  filter(Subtype %in% c("BRCA.LumA", "BRCA.LumB", "BRCA.Her2", "BRCA.Basal")) %>%
  dplyr::select(SampleID, Subtype) %>%
  mutate(SampleID = substr(SampleID, 1, 15)) %>%
  mutate(SampleID = gsub("\\.", "-", SampleID)) %>%
  mutate(Subtype = as.factor(Subtype)) %>% # mlr3 requires the classification target to be a factor
  distinct(SampleID, .keep_all = TRUE)

# 1.2 Prepare Feature Set A: Drug predictions (Your Model)
# Use the previously defined cytotoxic_drugs list
drug_features <- pred_full %>%
  filter(DrugName %in% cytotoxic_drugs) %>%
  dplyr::select(SampleID, DrugName, Predicted_IC50) %>%
  mutate(SampleID = substr(SampleID, 1, 15)) %>%
  mutate(SampleID = gsub("\\.", "-", SampleID)) %>%
  pivot_wider(names_from = DrugName, values_from = Predicted_IC50)

# 1.3 Prepare Feature Set B: Tumor Purity (Baseline)
purity_feature <- est_clean %>%
  dplyr::select(SampleID, TumorPurity) %>%
  mutate(SampleID = substr(SampleID, 1, 15)) %>%
  mutate(SampleID = gsub("\\.", "-", SampleID))

# 1.4 Build two independent datasets for modeling
# Dataset 1: Purity only
data_purity <- target_labels %>%
  inner_join(purity_feature, by = "SampleID") %>%
  dplyr::select(-SampleID) %>% # Remove ID column, keep only features and labels
  na.omit()

# Dataset 2: Drug predictions
data_comb <- target_labels %>%
  inner_join(drug_features, by = "SampleID") %>%
  dplyr::select(-SampleID) %>%
  # Use unnest() to expand all list-type columns
  unnest(cols = everything(), keep_empty = TRUE) %>% # Expand all columns
  na.omit()

print(paste("Samples in Purity Task:", nrow(data_purity)))
print(paste("Samples in Comb Task:", nrow(data_comb)))


# ==============================================================================
# 2. Define Tasks
# ==============================================================================

# Task 1: Baseline (Purity)
task_purity <- as_task_classif(data_purity, target = "Subtype", id = "Baseline (Purity)")

# Task 2: Your Model (Comb Drugs)
task_comb <- as_task_classif(data_comb, target = "Subtype", id = "Comb Model (Drug Scores)")

# Set stratified sampling (Stratification)
# Critical: ensures subtype proportions match in each fold
task_purity$col_roles$stratum <- "Subtype"
task_comb$col_roles$stratum <- "Subtype"

# ==============================================================================
# 3. Define Learner and Resampling strategy
# ==============================================================================

# Use Random Forest (ranger)
# importance = "impurity" for feature importance plots
learner <- lrn("classif.ranger", predict_type = "prob", importance = "impurity")

# 5-fold cross-validation
resampling <- rsmp("cv", folds = 5)

# ==============================================================================
# 4. Execute Benchmark (core comparison)
# ==============================================================================

# Build design matrix: two tasks, same learner, same resampling
design <- benchmark_grid(
  tasks = list(task_purity, task_comb),
  learners = learner,
  resamplings = resampling
)

# Start run (set random seed for reproducibility)
set.seed(42)
bmr <- benchmark(design)

# View aggregated results
print(bmr$aggregate(msr("classif.acc")))

# [Plot code removed -> see figures.qmd]
  labs(
    title = "Stratification accuracy",
    y = "ACC (5-fold CV)",
    x = ""
  ) +
  theme_classic() +
  theme(axis.text.x = element_text(face = "bold", size = 0))

# ==============================================================================
# 6. Draw confusion matrix heatmap for Comb model
# ==============================================================================

# Extract prediction results for Comb task
res_comb <- bmr$resample_result(2) # Index 2 corresponds to task_comb (based on grid order)
cm <- res_comb$prediction()$confusion

# Convert to data frame for ggplot plotting
cm_df <- as.data.frame(cm)
names(cm_df) <- c("True_Class", "Predicted_Class", "Freq")

# Plot
# [Plot code removed -> see figures.qmd]

# ==============================================================================
# 7. Feature importance analysis
# ==============================================================================

# Retrain on all data to extract importance
final_model <- lrn("classif.ranger", importance = "impurity")
final_model$train(task_comb)

# Extract importance and plot
importance_data <- as.data.frame(final_model$importance())
names(importance_data) <- "Importance"
importance_data$Feature <- rownames(importance_data)

# Take top 15
top_features <- importance_data %>%
  arrange(desc(Importance)) %>%
  head(15)

# [Plot code removed -> see figures.qmd]


####survival

# ==============================================================================
# 1. Modify data retrieval (add Age, Gender, Stage)
# ==============================================================================
get_survival_data_extended <- function(cancer_codes) {
  all_clin <- list()
  for (proj in cancer_codes) {
    proj_id <- paste0("TCGA-", proj)
    # Get clinical data
    clin <- GDCquery_clinic(project = proj_id, type = "clinical", save.csv = FALSE)
    
    # Check column existence (names vary across cancers)
    cols_to_select <- c("submitter_id", "vital_status", "days_to_death", 
                        "days_to_last_follow_up", "gender", "age_at_index", 
                        "ajcc_pathologic_stage")
    
    # Extract existing columns
    valid_cols <- intersect(cols_to_select, colnames(clin))
    
    clin_sub <- clin %>%
      select(all_of(valid_cols)) %>%
      mutate(CancerCode = proj)
    
    all_clin[[proj]] <- clin_sub
  }
  bind_rows(all_clin)
}

# Get data
my_cancers <- c("LGG", "BRCA", "PRAD", "LUAD", "SKCM") 
survival_raw <- get_survival_data_extended(my_cancers)

# ==============================================================================
# 2. Data cleaning (survival time, age, stage)
# ==============================================================================
survival_clean <- survival_raw %>%
  dplyr::rename(PatientID = submitter_id) %>%
  mutate(
    # --- A. Survival time ---
    OS_Status = ifelse(vital_status == "Dead", 1, 0),
    OS_Time = ifelse(OS_Status == 1, days_to_death, days_to_last_follow_up),
    OS_Time_Months = OS_Time / 30.4,
    
    # --- B. Age (ensure numeric) ---
    # Some use 'age_at_index', others 'age_at_diagnosis'
    Age = as.numeric(age_at_index), 
    
    # --- C. Stage simplification (Stage I-IV) ---
    # TCGA staging is messy (IIA, IIB); simplify uniformly
    Stage_Simple = case_when(
      grepl("Stage IV", ajcc_pathologic_stage, ignore.case = T) ~ "Stage IV",
      grepl("Stage III", ajcc_pathologic_stage, ignore.case = T) ~ "Stage III",
      grepl("Stage II", ajcc_pathologic_stage, ignore.case = T) ~ "Stage II",
      grepl("Stage I", ajcc_pathologic_stage, ignore.case = T) ~ "Stage I",
      TRUE ~ NA_character_ # Unknown or no staging
    )
  ) %>%
  filter(!is.na(OS_Time) & OS_Time > 0)

# ==============================================================================
# 3. Merge data (CBI + Clinical + Subtype)
# ==============================================================================
# Assumes cbi_df and clean_subtypes exist (from previous code)

surv_analysis_data <- cbi_df %>%
  mutate(PatientID = substr(SampleID, 1, 12)) %>%
  inner_join(survival_clean, by = "PatientID") %>%
  # Still recommended: add subtypes for subgroup analysis
  left_join(clean_subtypes %>% mutate(PatientID = substr(SampleID, 1, 12)), by = "PatientID") %>%
  distinct(PatientID, .keep_all = TRUE) # deduplicate

print("Data merging complete. Columns available:")
print(colnames(surv_analysis_data))


# ==============================================================================
# 4. KM plotting function (Optimal Cutoff specific)
# ==============================================================================
plot_km_optimal <- function(data, cancer_code) {
  
  # 1. Filter data
  df_sub <- data %>% filter(CancerCode.x == cancer_code)
  
  if(nrow(df_sub) < 20) return(NULL)
  
  # 2. Find optimal cutoff (MaxStat) 
  res.cut <- surv_cutpoint(df_sub, time = "OS_Time_Months", event = "OS_Status", 
                           variables = "CBI", minprop = 0.2)
  
  cutoff_val <- res.cut$cutpoint$cutpoint
  
  # 3. Classify by optimal cutoff (High vs Low)
  df_sub <- df_sub %>%
    mutate(Group = ifelse(CBI > cutoff_val, "High CBI", "Low CBI"))
  
  # 4. Fit curves
  fit <- survfit(Surv(OS_Time_Months, OS_Status) ~ Group, data = df_sub)
  
  # 5. Plot
  ggsurvplot(
    fit, 
    data = df_sub,
    pval = TRUE,             
    pval.method = TRUE,      # Display Log-rank
    risk.table = TRUE,       
    conf.int = FALSE,         
    palette = c("#DC143C", "#4682B4"), # High=Red, Low=Blue
    title = paste0("Optimal Cutoff Survival: ", cancer_code),
    subtitle = paste0("Cutoff = ", round(cutoff_val, 3), 
                      " (High n=", sum(df_sub$Group=="High CBI"), 
                      ", Low n=", sum(df_sub$Group=="Low CBI"), ")"),
    xlab = "Time (Months)",
    ylab = "Overall Survival Probability",
    ggtheme = theme_classic(),
    risk.table.height = 0.25
  )
}

# --- Execute plotting ---
# 1. LGG (expected very significant)
p_lgg <- plot_km_optimal(surv_analysis_data, "LGG")
# [Plot code removed -> see figures.qmd]

# 2. BRCA (check if Optimal can salvage)
p_brca <- plot_km_optimal(surv_analysis_data, "BRCA")
# [Plot code removed -> see figures.qmd]

p_skcm <- plot_km_optimal(surv_analysis_data, "SKCM")
# [Plot code removed -> see figures.qmd]


# ==============================================================================
# 5. Multivariate Cox regression & Forest Plot
# ==============================================================================

run_multivariate_cox <- function(data, cancer_code) {
  
  # 1. Filter and clean data for specific cancer
  df_cox <- data %>% 
    filter(CancerCode.x == cancer_code) %>%
    filter(!is.na(Age)) # remove missing age
  
  # 2. Build formula
  # Note: LGG usually has no Stage, only Grade (CDR may lack Grade column)
  # So we dynamically adjust formula per cancer type
  
  if (cancer_code == "LGG") {
    # LGG only has Age & Gender (or merge Grade data)
    formula_cox <- as.formula("Surv(OS_Time_Months, OS_Status) ~ CBI + Age + gender")
  } else {
    # BRCA etc. usually have Stage
    # Filter out samples with missing Stage
    df_cox <- df_cox %>% filter(!is.na(Stage_Simple))
    formula_cox <- as.formula("Surv(OS_Time_Months, OS_Status) ~ CBI + Age + gender + Stage_Simple")
  }
  
  print(paste("Running Cox for:", cancer_code, "with N =", nrow(df_cox)))
  
  # 3. Run Cox model
  res.cox <- coxph(formula_cox, data = df_cox)
  
  # Print summary
  print(summary(res.cox))
  
  # 4. Draw forest plot (Forest Model)
  # This is a nice publication-quality chart
  p_forest <- forest_model(
    res.cox,
    format_options = forest_model_format_options(
      text_size = 4,
      point_size = 3
    )
  ) + 
    ggtitle(paste0("Multivariate Cox Regression: ", cancer_code))
  
  return(p_forest)
}

# --- Execute Cox analysis ---

# 1. LGG forest plot
# Expected: CBI HR significantly > 1 (poor prognosis)
# Or < 1 (if CBI reflects chemo-sensitivity benefit)
forest_lgg <- run_multivariate_cox(surv_analysis_data, "LGG")
# [Plot code removed -> see figures.qmd]

# 2. BRCA Forest Plot
# Check if CBI becomes significant after adjusting for Stage
forest_brca <- run_multivariate_cox(surv_analysis_data, "BRCA")
# [Plot code removed -> see figures.qmd]




# ==============================================================================
# 1. Define drug groupings (Drug Sets)
# ==============================================================================
# Group drugs by mechanism, also keep "All"
drug_sets_list <- list(
  "All_Cytotoxic" = c("Vinblastine", "Cisplatin", "Cytarabine", "Docetaxel", "Methotrexate", 
                      "5-Fluorouracil", "Paclitaxel", "Irinotecan", "Oxaliplatin", 
                      "Temozolomide", "Epirubicin", "Cyclophosphamide", "Mitoxantrone", 
                      "Dactinomycin", "Bleomycin", "Dacarbazine", "Bleomycin (50 uM)"),
  
  "Microtubule_Inhibitors" = c("Vinblastine", "Docetaxel", "Paclitaxel"),
  
  "DNA_Damaging" = c("Cisplatin", "Oxaliplatin", "Temozolomide", "Cyclophosphamide", 
                     "Dacarbazine", "Bleomycin", "Bleomycin (50 uM)"),
  
  "Topoisomerase_Inhibitors" = c("Irinotecan", "Epirubicin", "Mitoxantrone", "Dactinomycin"),
  
  "Antimetabolites" = c("Cytarabine", "Methotrexate", "5-Fluorouracil")
)

# ==============================================================================
# 2. Define CBI computation core function (supports Mean, PCA, FA)
# ==============================================================================
calculate_cbi_variant <- function(pred_data, drug_list, method = "Mean") {
  
  # 1. Filter drugs
  subset_data <- pred_data %>%
    filter(DrugName %in% drug_list) %>%
    select(SampleID, DrugName, Predicted_IC50)
  
  # 2. Convert to wide format (Sample x Drug)
  wide_data <- subset_data %>%
    mutate(SampleID = substr(SampleID, 1, 15)) %>% # Normalize IDs
    # Handle duplicates: take the mean
    group_by(SampleID, DrugName) %>%
    summarise(Predicted_IC50 = mean(Predicted_IC50, na.rm=TRUE), .groups="drop") %>%
    pivot_wider(names_from = DrugName, values_from = Predicted_IC50) %>%
    tibble::column_to_rownames("SampleID") %>%
    na.omit() # PCA/FA cannot have missing values
  
  # If too few samples or drugs, cannot compute
  if (nrow(wide_data) < 10 || ncol(wide_data) < 2) return(NULL)
  
  # 3. Compute Score based on method
  scores <- tryCatch({
    if (method == "Mean") {
      # Z-score then average
      scaled_data <- scale(wide_data)
      rowMeans(scaled_data, na.rm = TRUE)
      
    } else if (method == "PCA") {
      # Take the first principal component (PC1)
      # PC1 usually captures the largest variance direction (the main "resistance/sensitivity" axis)
      pca_res <- prcomp(wide_data, center = TRUE, scale. = TRUE)
      pca_res$x[, 1] 
      
    } else if (method == "FA") {
      # Factor Analysis (FA), take the first factor
      # Compared to PCA, FA focuses more on extracting latent common factors
      fa_res <- factanal(wide_data, factors = 1, scores = "regression")
      fa_res$scores[, 1]
    }
  }, error = function(e) return(NULL))
  
  if (is.null(scores)) return(NULL)
  
  # 4. Format and return
  result <- data.frame(SampleID = names(scores), CBI = scores) %>%
    mutate(SampleID = gsub("\\.", "-", SampleID)) # Fix ID format
  
  return(result)
}

# ==============================================================================
# 3. Loop and evaluate
# ==============================================================================

# Set cancer types to test
target_cancers <- c("LGG", "BRCA", "PRAD", "LUAD", "SKCM") 

# Container for storing results
results_table <- data.frame()

# Loop: Cancer -> Drug Set -> Method
for (cancer in target_cancers) {
  
  # Get survival data for this cancer (using previously defined survival_clean)
  surv_cancer <- survival_clean %>% 
    filter(CancerCode == cancer) %>%
    mutate(SampleID_Join = substr(PatientID, 1, 12)) # Prepare ID for joining
  
  if(nrow(surv_cancer) < 20) next
  
  for (set_name in names(drug_sets_list)) {
    drugs <- drug_sets_list[[set_name]]
    
    for (method in c("Mean", "PCA", "FA")) {
      
      # A. Compute CBI
      # Note: pred_full should contain only this cancer's data, or compute globally then merge
      # For speed, it is recommended to first filter predictions for this cancer (if pred_full has CancerCode)
      # Here we assume pred_full contains all
      
      cbi_res <- calculate_cbi_variant(pred_full, drugs, method)
      
      if (is.null(cbi_res)) next
      
      # B. Merge survival data
      analysis_set <- cbi_res %>%
        mutate(SampleID_Join = substr(SampleID, 1, 12)) %>%
        inner_join(surv_cancer, by = "SampleID_Join")
      
      if (nrow(analysis_set) < 20) next
      
      # C. Run univariate Cox regression (Continuous Variable)
      # We test whether CBI as a continuous variable is significant
      cox_fit <- tryCatch({
        coxph(Surv(OS_Time_Months, OS_Status) ~ CBI, data = analysis_set)
      }, error = function(e) NULL)
      
      if (!is.null(cox_fit)) {
        tidy_res <- tidy(cox_fit)
        
        # D. Record results
        results_table <- rbind(results_table, data.frame(
          Cancer = cancer,
          Drug_Set = set_name,
          Method = method,
          HR = exp(tidy_res$estimate), # Hazard Ratio
          P_Value = tidy_res$p.value,
          C_Index = summary(cox_fit)$concordance[1], # Concordance index
          N_Samples = nrow(analysis_set)
        ))
      }
    }
  }
}

# ==============================================================================
# 4. Display best strategies (Best Performers)
# ==============================================================================

# Sort by P-value, view the best strategy for each cancer
best_strategies <- results_table %>%
  group_by(Cancer) %>%
  dplyr::filter(HR>1) 
#  arrange(P_Value) %>%
#  slice_head(n = 3) # Show top 3 for each cancer

print(best_strategies)


run_cbi_survival_analysis <- function(cancer_code, 
                                      drug_set_name, 
                                      method = "FA", 
                                      flip_sign = FALSE) {
  
  message(paste0(">>> Analyzing: ", cancer_code, " | Drug Set: ", drug_set_name, " | Method: ", method))
  
  # 1. Get drug list
  if (!drug_set_name %in% names(drug_sets_list)) {
    stop("Drug set name not found in 'drug_sets_list'.")
  }
  drugs <- drug_sets_list[[drug_set_name]]
  
  # 2. Compute CBI
  # Note: assumes calculate_cbi_variant and pred_full are already defined in the environment
  cbi_df <- calculate_cbi_variant(pred_full, drugs, method)
  
  if (is.null(cbi_df)) {
    warning("CBI calculation failed (not enough data/drugs).")
    return(NULL)
  }
  
  # 3. Sign flip (for cases where PCA/FA direction is ambiguous)
  if (flip_sign) {
    message("Note: Flipping CBI sign (-CBI).")
    cbi_df$CBI <- -cbi_df$CBI
  }
  
  # 4. Data merging and cleaning
  # Assumes survival_clean is already in the environment
  plot_data <- cbi_df %>%
    mutate(SampleID_Join = substr(SampleID, 1, 12)) %>%
    inner_join(survival_clean, by = c("SampleID_Join" = "PatientID")) %>%
    mutate(CancerCode.x = CancerCode) # To fit the previously defined plotting function
  
  # Check sample size
  n_samples <- nrow(plot_data %>% filter(CancerCode.x == cancer_code))
  if (n_samples < 20) {
    warning(paste("Not enough samples for", cancer_code))
    return(NULL)
  }
  
  # 5. Generate charts
  # A. KM Curve (Optimal Cutoff)
  # Assumes plot_km_optimal is already in the environment
  p_km <- plot_km_optimal(plot_data, cancer_code)
  
  # B. Cox Forest Plot
  # Assumes run_multivariate_cox is already in the environment
  p_forest <- run_multivariate_cox(plot_data, cancer_code)
  
  # 6. Return result list
  return(list(
    data = plot_data,
    plot_km = p_km,
    plot_forest = p_forest
  ))
}

res_lgg <- run_cbi_survival_analysis(
  cancer_code = "LGG",
  drug_set_name = "Topoisomerase_Inhibitors",
  method = "FA",
  flip_sign = TRUE 
)

res_skcm <- run_cbi_survival_analysis(
  cancer_code = "SKCM",
  drug_set_name = "Microtubule_Inhibitors",
  method = "Mean",
  flip_sign = FALSE
)

res_brca <- run_cbi_survival_analysis(
  cancer_code = "BRCA",
  drug_set_name = "Antimetabolites",
  method = "FA",
  flip_sign = FALSE
)

##### Discovery of subtype -specific survival analysis in BRCA
# ==============================================================================
# 1. Prepare data
# ==============================================================================

# A. Compute CBI
# Microtubule inhibitors (taxanes) are the cornerstone of breast cancer chemotherapy
cbi_brca <- calculate_cbi_variant(pred_full, drug_sets_list[["Antimetabolites"]], "FA")

# B. Check direction (IMPORTANT!)
# Assume High CBI = High Proliferation = Aggressive
# If PCA direction is reversed, remember flip_sign = TRUE (here we assume no flip, adjust based on Forest Plot)
# cbi_brca$CBI <- -cbi_brca$CBI 

# C. Merge survival and subtype data
plot_data_brca <- cbi_brca %>%
  mutate(SampleID_Join = substr(SampleID, 1, 12)) %>%
  inner_join(survival_clean, by = c("SampleID_Join" = "PatientID")) %>%
  filter(CancerCode== "BRCA") %>%
  inner_join(clean_subtypes %>% mutate(SampleID_12 = substr(SampleID,1,12)), by = c("SampleID_Join" = "SampleID_12"))# with Subtype


# ==============================================================================
# 2. Define subtype-specific plotting function
# ==============================================================================
run_subtype_analysis <- function(data, target_subtype) {
  
  # Filter specific subtype
  df_sub <- data %>% 
    filter(Subtype == target_subtype)
  
  # Check sample size
  if(nrow(df_sub) < 30) {
    message(paste("Not enough samples for", target_subtype))
    return(NULL)
  }
  
  # --- KM Curve (Optimal Cutoff) ---
  res.cut <- surv_cutpoint(df_sub, time = "OS_Time_Months", event = "OS_Status", 
                           variables = "CBI", minprop = 0.2)
  cutoff_val <- res.cut$cutpoint$cutpoint
  
  df_sub <- df_sub %>%
    mutate(Group = ifelse(CBI > cutoff_val, "High CBI", "Low CBI"))
  
  fit <- survfit(Surv(OS_Time_Months, OS_Status) ~ Group, data = df_sub)
  
  p_km=ggsurvplot(
    fit, 
    data = df_sub,
    pval = TRUE,             
    pval.method = TRUE,      # Display Log-rank
    risk.table = TRUE,       
    conf.int = FALSE,         
    palette = c("#DC143C", "#4682B4"), # High=Red, Low=Blue
    title = paste0("Optimal Cutoff Survival: ", target_subtype),
    subtitle = paste0("Cutoff = ", round(cutoff_val, 3), 
                      " (High n=", sum(df_sub$Group=="High CBI"), 
                      ", Low n=", sum(df_sub$Group=="Low CBI"), ")"),
    xlab = "Time (Months)",
    ylab = "Survival Probability",
    ggtheme = theme_classic(),
    risk.table.height = 0.25
  )
  # --- Cox regression ---
  df_sub <- df_sub %>% filter(!is.na(Stage_Simple)) %>% filter(!is.na(gender))
  
  cox_res <- coxph(Surv(OS_Time_Months, OS_Status) ~ CBI + Age + gender + Stage_Simple, data = df_sub)
  p_forest <- forest_model(
    cox_res,
    format_options = forest_model_format_options(
      text_size = 4,
      point_size = 3
    )
  ) + 
    ggtitle(paste0("Multivariate Cox Regression: ",target_subtype))
  
  return(list(km = p_km, cox = cox_res, forest = p_forest))
}

# ==============================================================================
# 3. Execute analysis
# ==============================================================================

# --- A. Basal-like (TNBC, main chemo population) ---
# Expected: most likely significant. High CBI may represent hyperproliferation/refractory, or chemo-sensitive benefit.
res_basal <- run_subtype_analysis(plot_data_brca, "BRCA.Basal")
print(res_basal$km)
print(res_basal$forest)
print(summary(res_basal$cox))

# --- B. Luminal B (high-grade HR+, often needs chemo) ---
# Expected: High CBI indicates poor prognosis.
res_lumb <- run_subtype_analysis(plot_data_brca, "BRCA.LumB")
print(res_lumb$km)

#####

lgg_subtypes <- TCGAbiolinks::PanCancerAtlas_subtypes() %>%
  filter(cancer.type == "LGG") %>%
  select(
    SampleID = pan.samplesID,
    Subtype_DNAmeth # This is the core subtype
  ) %>%
  mutate(SampleID = substr(SampleID, 1, 12)) %>%
  # Filter out NA subtypes
  filter(!is.na(Subtype_DNAmeth) & Subtype_DNAmeth != "NA")

common_lgg_samples <- intersect(unique(substr(pred_full$SampleID, 1, 12)), lgg_subtypes$SampleID)
#transcriptome = read_delim("/Users/hechang/Documents/chenlab/CellHit_onlyCPU/data/transcriptomics/celligner_CCLE_TCGA.csv")
top2a_df = transcriptome[match(common_lgg_samples,(transcriptome$index %>% substr(.,1,12))), c('index', 'TOP2A')] %>%
  mutate(SampleID_12 = substr(index, 1, 12))

best_cbi_df = calculate_cbi_variant(pred_full,drug_sets_list[["Topoisomerase_Inhibitors"]], "FA") %>%
  mutate(CBI = -as.numeric(CBI))

lgg_master_data <- best_cbi_df %>%
  mutate(SampleID_12 = substr(SampleID, 1, 12)) %>%
  # 1. Merge subtypes
  inner_join(lgg_subtypes, by = c("SampleID_12" = "SampleID")) %>%
  # 2. Merge survival (for heatmap annotation)
  inner_join(survival_clean, by = c("SampleID_12" = "PatientID")) %>%
  left_join(top2a_df)


# 1. For visual appeal, sort subtypes by median CBI
lgg_master_data$Subtype_DNAmeth <- reorder(
  lgg_master_data$Subtype_DNAmeth, 
  lgg_master_data$CBI, 
  FUN = median
)

# 2. Plot
# [Plot code removed -> see figures.qmd]
  # Statistical test: compare lowest vs highest group
  stat_compare_means(
    method = "wilcox.test", 
    label = "p.signif",
    ref.group = levels(lgg_master_data$Subtype_DNAmeth)[1], # Use the group with lowest CBI as reference
    label.y = max(lgg_master_data$CBI) * 1.05
  ) +
  # Global P-value
  stat_compare_means(label.y = max(lgg_master_data$CBI) * 1.2, size = 4) +
  
  # Color: Use Magma, darker = higher score (more malignant)
  scale_fill_viridis_d(option = "magma", direction = -1) +
  
  labs(
    title = "Subtype Stratification",
    x = "DNA Methylation Subtype",
    y = "Topoisomerase CBI (Factor Analysis)"
  ) +
  theme_classic() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(face = "bold", angle = 30, hjust = 1),
    plot.title = element_text(face = "bold")
  )

# [Plot code removed -> see figures.qmd]


# ==============================================================================
# Panel D: Mechanism Validation (Target Correlation)
# ==============================================================================

# [Plot code removed -> see figures.qmd]
  geom_point(aes(color = Subtype_DNAmeth), alpha = 0.7, size = 2) +
  
  # Global regression line
  geom_smooth(method = "lm", color = "black", linetype = "dashed", fill = "grey80") +
  
  # Correlation coefficient
  stat_cor(method = "pearson", label.x.npc = "left", label.y.npc = "top", size = 5) +
  
  scale_color_viridis_d(option = "magma", direction = -1, name = "Subtype") +
  
  labs(
    title = "Target Engagement",
    x = "TOP2A Expression (log2 TPM)",
    y = "CBI Score"
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "right"
  )

# [Plot code removed -> see figures.qmd]


# [Plot code removed -> see figures.qmd]
  geom_point(alpha = 0.8, size = 1.5) +
  
  # 2. Regression line: fitted per facet
  geom_smooth(method = "lm", color = "black", linetype = "dashed", se = TRUE) +
  
  # 3. Correlation statistics
  stat_cor(method = "pearson", label.x.npc = "left", label.y.npc = "top", size = 3.5) +
  
  # 4. Key: faceted display
  facet_wrap(~Subtype_DNAmeth, scales = "free") + 
  
  scale_color_viridis_d(option = "magma", direction = -1) +
  
  labs(
    title = "Target Engagement: Subtype-Specific Analysis",
    x = "TOP2A Expression (log2 TPM)",
    y = "CBI Score"
  ) +
  theme_bw() + # Use bw theme for better faceted visuals
  theme(
    strip.background = element_rect(fill = "grey90"), # Facet strip background
    strip.text = element_text(face = "bold"),         # Facet strip text
    legend.position = "none"                          # Facets already have titles, legend is redundant
  )

# [Plot code removed -> see figures.qmd]


#crispr_data = read_delim("/Users/hechang/Documents/chenlab/data/23Q4/CRISPRGeneDependency.csv")
#colnames(crispr_data)[1] = ''

####drug comb
drug_comb_nci = read_delim("/Users/hechang/Documents/chenlab/data/drug_comb_nci.csv")



# ==============================================================================
# 1. Build name mapping dictionary (Name Map)
# ==============================================================================
# Left side = NCI names, right side = names in pred_full
name_map <- c(
  # --- Direct Matches ---
  "Cyclophosphamide" = "Cyclophosphamide",
  "Fluorouracil"     = "5-Fluorouracil",
  "5-Fluorouracil"   = "5-Fluorouracil",
  "Methotrexate"     = "Methotrexate",
  "Epirubicin"       = "Epirubicin",
  "Docetaxel"        = "Docetaxel",
  "Oxaliplatin"      = "Oxaliplatin",
  "Irinotecan"       = "Irinotecan",
  "Paclitaxel"       = "Paclitaxel",
  "Cisplatin"        = "Cisplatin",
  "Vinblastine"      = "Vinblastine",
  "Bleomycin"        = "Bleomycin",
  "Dacarbazine"      = "Dacarbazine",
  "Mitoxantrone"     = "Mitoxantrone",
  "Dactinomycin"     = "Dactinomycin",
  "Cytarabine"       = "Cytarabine",
  "Temozolomide"     = "Temozolomide",
  
  # --- Proxies (pharmacological substitutes - critical step) ---
  "Doxorubicin"      = "Epirubicin",       # Anthracycline substitute
  "Capecitabine"     = "5-Fluorouracil",   # 5-FU prodrug
  "Carboplatin"      = "Cisplatin",        # Platinum substitute
  "Vincristine"      = "Vinblastine",      # Vinca alkaloid substitute
  "Ifosfamide"       = "Cyclophosphamide", # Alkylating agent substitute
  "Daunorubicin"     = "Epirubicin",       # Anthracycline substitute
  
  # --- Missing / No Prediction (drugs without model coverage) ---
  # These drugs are not in your model and have no good substitute, set to NA
  "Leucovorin"       = NA, # Adjuvant drug
  "Bevacizumab"      = NA, # Antibody
  "Cetuximab"        = NA, # Antibody
  "Etoposide"        = NA, # Etoposide is not in your list, no good substitute (Top2 inhibitor is not equivalent)
  "Prednisone"       = NA, # Hormone
  "Procarbazine"     = NA,
  "Mechlorethamine"  = NA,
  "Bortezomib"       = NA,
  "Dexamethasone"    = NA,
  "Melphalan"        = NA,
  "Busulfan"         = NA,
  'Gemcitabine'      = NA
)

# ==============================================================================
# 2. Clean combination data and compute Regimen Score
# ==============================================================================

# 2.1 Explode the NCI table
comb_processed <- drug_comb_nci %>%
  # Split by semicolons
  separate_rows(Drugs, sep = ";") %>%
  mutate(Drugs_Clean = str_trim(Drugs)) %>%
  # Map names
  mutate(Model_Drug = name_map[Drugs_Clean]) %>%
  # Filter out drugs that cannot be predicted
  filter(!is.na(Model_Drug)) %>%
  # Reconfirm that mapped drugs are actually in your prediction results
  filter(Model_Drug %in% unique(pred_full$DrugName))

# Check how many drugs were retained
print("Mapped Drugs Coverage:")
print(table(comb_processed$Model_Drug))


# Define mapping function
assign_cancer_code <- function(disease_name) {
  case_when(
    # --- TCGA Standard Cancers ---
    disease_name == "breast invasive carcinoma" ~ "BRCA",
    disease_name == "lung adenocarcinoma" ~ "LUAD",
    disease_name == "lung squamous cell carcinoma" ~ "LUSC",
    disease_name == "colon adenocarcinoma" ~ "COAD",
    disease_name == "rectum adenocarcinoma" ~ "READ",
    disease_name == "prostate adenocarcinoma" ~ "PRAD",
    disease_name == "bladder urothelial carcinoma" ~ "BLCA",
    disease_name == "liver hepatocellular carcinoma" | disease_name == "hepatocellular carcinoma" ~ "LIHC",
    disease_name == "stomach adenocarcinoma" ~ "STAD",
    disease_name == "skin cutaneous melanoma" | disease_name == "melanoma" ~ "SKCM",
    disease_name == "glioblastoma multiforme" ~ "GBM",
    disease_name == "brain lower grade glioma" | disease_name == "glioma" ~ "LGG",
    disease_name == "acute myeloid leukemia" ~ "LAML",
    disease_name == "thyroid carcinoma" ~ "THCA",
    disease_name == "ovarian serous cystadenocarcinoma" ~ "OV",
    disease_name == "pancreatic adenocarcinoma" ~ "PAAD",
    disease_name == "kidney clear cell carcinoma" ~ "KIRC",
    disease_name == "kidney papillary cell carcinoma" ~ "KIRP",
    disease_name == "kidney chromophobe" ~ "KICH",
    disease_name == "uterine corpus endometrioid carcinoma" ~ "UCEC",
    disease_name == "cervical & endocervical cancer" ~ "CESC",
    disease_name == "head & neck squamous cell carcinoma" ~ "HNSC",
    disease_name == "mesothelioma" ~ "MESO",
    disease_name == "thymoma" ~ "THYM",
    disease_name == "testicular germ cell tumor" ~ "TGCT",
    disease_name == "esophageal carcinoma" ~ "ESCA",
    disease_name == "pheochromocytoma & paraganglioma" ~ "PCPG",
    disease_name == "diffuse large B-cell lymphoma" | disease_name == "lymphoma" ~ "DLBC",
    disease_name == "uveal melanoma" ~ "UVM",
    disease_name == "adrenocortical carcinoma" | disease_name == "adrenocortical cancer" ~ "ACC",
    disease_name == "cholangiocarcinoma" ~ "CHOL",
    disease_name == "uterine carcinosarcoma" ~ "UCS",
    
    # --- TARGET / Pediatric / Non-TCGA Cancers ---
    str_detect(disease_name, "neuroblastoma") ~ "NBL",
    str_detect(disease_name, "medulloblastoma") ~ "MB",
    str_detect(disease_name, "wilms tumor") ~ "WT",
    str_detect(disease_name, "osteosarcoma") ~ "OS",
    str_detect(disease_name, "rhabdomyosarcoma") ~ "RMS", # incl. alveolar/embryonal
    str_detect(disease_name, "acute lymphoblastic leukemia") ~ "ALL",
    str_detect(disease_name, "ewing sarcoma") ~ "EWS",
    str_detect(disease_name, "hepatoblastoma") ~ "HB",
    str_detect(disease_name, "retinoblastoma") ~ "RB",
    str_detect(disease_name, "ependymoma") ~ "EPN",
    str_detect(disease_name, "meningioma") ~ "MNG",
    
    # --- Sarcomas (General) ---
    str_detect(disease_name, "sarcoma") ~ "SARC", # Catch remaining sarcomas
    
    # --- Fallback ---
    TRUE ~ "Other" # Unrecognized types classified as Other
  )
}

# 1. Refine tcga_md, add CancerCode
tcga_md_clean <- tcga_md %>%
  mutate(CancerCode = assign_cancer_code(disease)) %>%
  select(th_dataset_id, disease, CancerCode) # Keep only needed columns

# Check mapping results for excessive "Other" entries
print("Distribution of Cancer Codes:")
print(table(tcga_md_clean$CancerCode))

# 2. Update pred_scaled generation logic (all samples)
# Note: ensure SampleID in pred_full matches th_dataset_id in tcga_md
# Assume pred_full$SampleID equals th_dataset_id (or needs minor processing)

pred_scaled_all <- pred_full %>%
  inner_join(tcga_md_clean, by = c("SampleID" = "th_dataset_id")) %>%
  group_by(DrugName) %>%
  mutate(Z_Score = scale(Predicted_IC50)) %>%
  ungroup() %>%
  select(SampleID, DrugName, Z_Score, CancerCode, disease)

print(paste("Original rows:", nrow(pred_full)))
print(paste("Merged rows:", nrow(pred_scaled_all)))

# Recalculate Regimen Score
regimen_scores_all <- comb_processed %>% # This is the previously prepared NCI regimen-to-single-drug mapping table
  inner_join(pred_scaled_all, by = c("Model_Drug" = "DrugName")) %>%
  
  group_by(SampleID, Name, Indication, CancerCode) %>%
  summarise(
    Regimen_Score = mean(Z_Score, na.rm = TRUE),
    Drug_Count = n_distinct(Model_Drug),
    .groups = "drop"
  ) %>%
  filter(Drug_Count >= 2)

# Check current cancer type coverage (should be much higher than before)
print(table(regimen_scores_all$CancerCode))

# ==============================================================================
# Panel A: Co-sensitivity of Approved Pairs
# ==============================================================================
# We choose two classic combinations:
# 1. BRCA: AC regimen (Doxorubicin + Cyclophosphamide) -> mapped as Epirubicin + Cyclophosphamide
# 2. COAD: FOLFOX regimen (5-FU + Oxaliplatin)


# 1. Define NCI Indication -> TCGA CancerCode mapping table
# Adjust based on your drug_comb_nci content
indication_map <- list(
  "Breast Cancer" = "BRCA",
  "Colorectal Cancer" = c("COAD", "READ"), # Colorectal cancers are usually combined
  "Lung Cancer" = c("LUAD", "LUSC"),       # Lung cancer merges adenocarcinoma and squamous
  "Ovarian Cancer" = "OV",
  "Testicular Cancer" = "TGCT",
  "Gastric Cancer" = "STAD",
  "Pancreatic Cancer" = "PAAD",
  "Hodgkin Lymphoma" = "DLBC", # TCGA only has DLBC, roughly corresponding
  "Non-Hodgkin Lymphoma" = "DLBC",
  "Urothelial Cancer" = "BLCA",
  "Neuroblastoma" = NA, # TCGA has no such cancer type
  "Soft Tissue Sarcoma" = "SARC",
  "Myeloproliferative Neoplasms" = "LAML" # Corresponds to leukemia
)

# 2. Prepare data for loop computation
# We need to get regimens and their drugs from comb_processed
regimens_to_test <- comb_processed %>%
  select(Name, Indication, Model_Drug) %>%
  distinct()

# Ensure only regimens with corresponding TCGA data are kept
regimens_to_test <- regimens_to_test %>%
  rowwise() %>%
  mutate(Target_Cancers = list(indication_map[[Indication]])) %>%
  filter(!any(is.na(Target_Cancers))) %>%
  ungroup()

print(head(regimens_to_test))

results_df <- data.frame()

# Get all regimen names
unique_regimens <- unique(regimens_to_test$Name)

for (reg_name in unique_regimens) {
  
  # A. Get regimen info
  sub_data <- regimens_to_test %>% filter(Name == reg_name)
  drugs <- unique(sub_data$Model_Drug)
  cancers <- unlist(unique(sub_data$Target_Cancers)) # May have multiple (e.g., LUAD, LUSC)
  
  # At least 2 drugs are required to compute correlation
  if (length(drugs) < 2) next
  
  # B. Generate all possible pairwise combinations
  drug_pairs <- combn(drugs, 2, simplify = FALSE)
  
  # C. Compute correlations in target cancer types
  for (cancer in cancers) {
    
    # Extract prediction data for this cancer
    cancer_pred <- pred_scaled_all %>%
      filter(CancerCode == cancer) %>%
      filter(DrugName %in% drugs) %>%
      select(SampleID, DrugName, Z_Score) %>%
      pivot_wider(names_from = DrugName, values_from = Z_Score)
    
    # If too few samples, skip
    if (nrow(cancer_pred) < 20) next
    
    # Compute R and P for each drug pair
    for (pair in drug_pairs) {
      d1 <- pair[1]
      d2 <- pair[2]
      
      # Check if columns exist
      if (!d1 %in% colnames(cancer_pred) || !d2 %in% colnames(cancer_pred)) next
      
      # Compute test
      test <- cor.test(cancer_pred[[d1]], cancer_pred[[d2]], method = "pearson")
      
      # Record result
      results_df <- rbind(results_df, data.frame(
        Regimen = reg_name,
        Cancer = cancer,
        Drug1 = d1,
        Drug2 = d2,
        R = as.numeric(test$estimate),
        P = as.numeric(test$p.value)
      ))
    }
  }
}

# D. Filter best results
# Rule: P < 0.05, sort by R descending
top_results <- results_df %>%
  filter(P < 0.05) %>%
  arrange(desc(R)) %>%
  # Optional: dedup, prevent one regimen from dominating; only keep the best pair per regimen
  group_by(Regimen) %>%
  slice_head(n = 1) %>%
  ungroup() %>%
  # Sort by R again
  arrange(desc(R))

print("Top 5 Best Correlated Pairs found:")
print(top_results, n=40)


best_4 <- top_results[c(1,11,12,16),]

plot_list_a <- list()

if (exists("best_4") && nrow(best_4) > 0) {
  plot_counter <- 1
  
  for (i in 1:nrow(best_4)) {
    
    info <- best_4[i, ]
    
    plot_dat <- pred_scaled_all %>%
      filter(CancerCode == info$Cancer) %>%
      filter(DrugName %in% c(info$Drug1, info$Drug2)) %>%
      select(SampleID, DrugName, Z_Score) %>%
      pivot_wider(names_from = DrugName, values_from = Z_Score) %>%
      na.omit()
    
    # After pivot_wider, if a drug has no data, the column may not be generated at all
    if (nrow(plot_dat) < 1) {
      message(paste("Skipping:", info$Regimen, "in", info$Cancer, "- Not enough matching samples."))
      next
    }
    
    if (!all(c(info$Drug1, info$Drug2) %in% colnames(plot_dat))) {
      message(paste("Skipping:", info$Regimen, "- One of the drugs is missing in the column names."))
      next
    }
    
    # 3. Plot: use .data[[string]] syntax instead of aes_string
    # This syntax handles names with hyphens like "5-Fluorouracil" perfectly, without object-not-found errors
# [Plot code removed -> see figures.qmd]
      # Add statistics
      stat_cor(method = "pearson", label.x.npc = "left", label.y.npc = "top", size = 4.5) +
      
      labs(
        title = paste0(info$Regimen, " Regimen"),
        subtitle = paste0(info$Cancer),
        x = info$Drug1,
        y = info$Drug2
      ) +
      theme_classic() +
      theme(
        plot.title = element_text(face = "bold", size = 12)
      )
    
    # Save to list
    plot_list_a[[plot_counter]] <- p
    plot_counter <- plot_counter + 1
  }
  
  # 4. Combined display
  if (length(plot_list_a) > 0) {
    # [Plot code removed -> see figures.qmd]
    # final_panel_a <- ggarrange(plotlist = plot_list_a, ...)
    # print(final_panel_a)
  } else {
    print("No valid plots were generated.")
  }
  
} else {
  print("No significant pairs found in 'best_4'.")
  # final_panel_a <- NULL
}




# ===============================
# Panel B
#

# 1. Prepare data: FOLFOX in COAD
target_regimen <- c("5-Fluorouracil", "Oxaliplatin")
target_cancer <- "COAD"

waterfall_data <- pred_scaled_all %>%
  filter(CancerCode == target_cancer) %>%
  filter(DrugName %in% target_regimen) %>%
  select(SampleID, DrugName, Z_Score)

# 2. Compute total score per patient for ordering
rank_data <- waterfall_data %>%
  group_by(SampleID) %>%
  summarise(Total_Score = sum(Z_Score)) %>%
  arrange(desc(Total_Score))

# 3. Set factor order (for waterfall style)
waterfall_data$SampleID <- factor(waterfall_data$SampleID, levels = rank_data$SampleID)

# 4. Plot
# [Plot code removed -> see figures.qmd]
  scale_fill_manual(values = c("5-Fluorouracil" = "#4682B4", "Oxaliplatin" = "#DC143C")) +
  
  # Add line representing sensitivity threshold (e.g., Z > 0)
  geom_hline(yintercept = 0, linetype="dashed", color="grey") +
  
  labs(
    title = "FOLFOX Response Stratification (COAD)",
    subtitle = "Identifying patients with dual sensitivity",
    x = "Patients",
    y = "Predicted Sensitivity (Z-Score)"
  ) +
  theme_classic() +
  theme(
    axis.text.x = element_blank(), # Hide x-axis patient IDs
    axis.ticks.x = element_blank(),
    legend.position = "top"
  )

# [Plot code removed -> see figures.qmd]


# ==============================================================================
# Panel C: Regimen Specificity Heatmap
# ==============================================================================

# 1. Compute mean score per regimen per cancer type
spec_data <- regimen_scores_all %>%
  group_by(Name, CancerCode) %>%
  summarise(Mean_Score = mean(Regimen_Score), .groups = "drop") %>%
  # Filter cancers/regimens with too few samples to keep the chart clean
  group_by(CancerCode) %>%
  filter(n() > 3) %>% 
  ungroup()

# 2. Convert to matrix
spec_mat <- spec_data %>%
  pivot_wider(names_from = CancerCode, values_from = Mean_Score) %>%
  tibble::column_to_rownames("Name") %>%
  as.matrix()

# Simple NA fill (use min value to avoid heatmap errors)
spec_mat[is.na(spec_mat)] <- min(spec_mat, na.rm = TRUE)

# 3. Draw heatmap
# We want to see diagonal trend
# [Plot code removed -> see figures.qmd]


# Set Top N (e.g., Top 600, or Top 10% of patients)
TOP_N <- 600

# ==============================================================================
# 1. Extract Top N sensitive patients per drug
# ==============================================================================
# Assume High Z-Score = Sensitive (reversed prediction logic)
top_patients_list <- pred_scaled_all %>%
  group_by(DrugName) %>%
  # For each drug, select the 600 patients with the highest Z-score
  slice_max(order_by = Z_Score, n = TOP_N) %>%
  ungroup() %>%
  select(DrugName, SampleID)

# ==============================================================================
# 2. Compute pairwise patient overlap between drugs (Intersection Matrix)
# ==============================================================================
# Use matrix multiplication for fast overlap computation (A x A^T)
# 1. Convert to 0/1 matrix (rows=patients, cols=drugs)
binary_mat <- table(top_patients_list$SampleID, top_patients_list$DrugName)
class(binary_mat) <- "matrix"
# Ensure binary (slice_max should guarantee uniqueness, but just in case)
binary_mat[binary_mat > 0] <- 1 

# 2. Matrix multiplication to get overlap matrix
# intersection_mat[i, j] = number of patients shared by drug i and drug j
intersection_mat <- t(binary_mat) %*% binary_mat

# 3. Convert to long-format data frame
overlap_df <- as.data.frame(as.table(intersection_mat)) %>%
  rename(Drug1 = Var1, Drug2 = Var2, Overlap_Count = Freq) %>%
  # Remove self-correlation
  filter(Drug1 != Drug2) %>%
  # Remove duplicate pairs (keep only half)
  filter(as.character(Drug1) < as.character(Drug2)) %>%
  # Filter out pairs with too low overlap (optional, e.g., skip if < 50 patients)
  filter(Overlap_Count > 100)

print(head(overlap_df))
# ==============================================================================
# 3. Build annotation dictionary (Support Level)
# ==============================================================================

# 3.1 Prepare Approved Pairs
# Reuse previous logic
approved_pairs_list <- comb_processed %>%
  select(Name, Model_Drug) %>%
  distinct() %>%
  inner_join(., ., by = "Name") %>%
  filter(Model_Drug.x < Model_Drug.y) %>%
  mutate(Pair_ID = paste(Model_Drug.x, Model_Drug.y, sep = "_")) %>%
  pull(Pair_ID) %>%
  unique()

# 3.2 Prepare Drug Indications mapping
# Logic: if a drug appears in a regimen for "Breast Cancer", it gets the "Breast Cancer" label
drug_indications_map <- comb_processed %>%
  select(Model_Drug, Indication) %>%
  distinct() %>%
  group_by(Model_Drug) %>%
  summarise(Indications = list(unique(Indication)))

# 3.3 Label overlap_df
# This is a row-by-row judgment process
get_support_level <- function(d1, d2) {
  pair_id <- paste(d1, d2, sep = "_")
  
  # 1. Check if Approved Combination
  if (pair_id %in% approved_pairs_list) {
    return("Approved Combination")
  }
  
  # 2. Check if Shared Indication
  # Get the indication lists for d1 and d2
  ind1 <- drug_indications_map$Indications[drug_indications_map$Model_Drug == d1]
  ind2 <- drug_indications_map$Indications[drug_indications_map$Model_Drug == d2]
  
  if (length(ind1) > 0 && length(ind2) > 0) {
    # Take intersection
    common <- intersect(unlist(ind1), unlist(ind2))
    if (length(common) > 0) {
      return("Sharing Indication")
    }
  }
  
  # 3. Neither a combination nor a shared indication
  return("Novel / Others")
}

# Apply function (vectorized via mapply)
overlap_df$Support_Level <- mapply(get_support_level, overlap_df$Drug1, overlap_df$Drug2)

# Set factor order so Approved appears topmost when plotting
overlap_df$Support_Level <- factor(overlap_df$Support_Level, 
                                   levels = c("Novel / Others", "Sharing Indication", "Approved Combination"))

print(table(overlap_df$Support_Level))

novel_subset <- overlap_df %>% filter(Support_Level == "Novel / Others")
threshold <- quantile(novel_subset$Overlap_Count, 0.90) 

print(paste("Novel Threshold (Top 10% overlap count):", threshold))

# Update Support_Level logic
overlap_df_refined <- overlap_df %>%
  mutate(Refined_Status = case_when(
    Support_Level == "Approved Combination" ~ "Approved Combination",
    Support_Level == "Sharing Indication" ~ "Sharing Indication",
    # If Novel and overlap count exceeds threshold -> define as Novel (High Potential)
    Support_Level == "Novel / Others" & Overlap_Count >= threshold ~ "Novel (High Overlap)",
    # Otherwise -> Others
    TRUE ~ "Others"
  ))

# Set factor order (determines legend order and plot layer order)
# We want Others at the bottom, Approved at the top
order_levels <- c("Others", "Novel (High Overlap)", "Sharing Indication", "Approved Combination")
overlap_df_refined$Refined_Status <- factor(overlap_df_refined$Refined_Status, levels = order_levels)
# ==============================================================================
# 4. Plot (Refined Bubble Plot)
# ==============================================================================

# Define new color scheme
custom_colors_refined <- c(
  "Approved Combination" = "#EE442F",  # Red (validation)
  "Sharing Indication"   = "#006400",  # Dark green (shared indication)
  "Novel (High Overlap)" = "#4A4A4A",  # Dark gray/black (discovery - highlight!)
  "Others"               = "#E0E0E0"   # Very light gray (background noise)
)

# Define alpha (make Others nearly invisible, highlight key points)
custom_alpha_refined <- c(
  "Approved Combination" = 1,
  "Sharing Indication"   = 0.8,
  "Novel (High Overlap)" = 0.9,
  "Others"               = 0.3 
)

# Sorting: ensure large bubbles don't obscure key points, Approved on top
# Logic: sort by factor level so Others is drawn first
overlap_df_refined <- overlap_df_refined %>% arrange(Refined_Status)

# [Plot code removed -> see figures.qmd]
  geom_tile(fill = NA, color = "white") + # Slightly separate with white
  
  # 2. Bubbles
  geom_point(aes(size = Overlap_Count, 
                 color = Refined_Status, 
                 alpha = Refined_Status)) +
  
  # 3. Color mapping
  scale_color_manual(values = custom_colors_refined) +
  
  # 4. Alpha mapping
  scale_alpha_manual(values = custom_alpha_refined) +
  
  # 5. Size mapping
  scale_size_continuous(range = c(1, 12), name = "Sample Overlap") +
  
  # 6. Theme adjustments
  theme_minimal() +
  theme(
    # X-axis text (top)
    axis.text.x.top = element_text(angle = 45, hjust = 0, vjust = 0, size = 9, face = "bold"),
    axis.text.y = element_text(size = 9, face = "bold"),
    axis.title = element_blank(),
    
    # Legend optimization
    legend.position = "right",
    legend.title = element_text(face = "bold"),
    panel.grid = element_blank(),
    
    # Add some margin to prevent text clipping
    plot.margin = margin(t = 10, r = 10, b = 10, l = 10)
  ) +
  scale_x_discrete(position = "top") +
  coord_fixed() +
  
  # Title
  labs(title = "Drug Combination Discovery")

# [Plot code removed -> see figures.qmd]


##### Bag-of-Words Analysis
target_samples = lgg_master_data$SampleID
res.cut <- surv_cutpoint(lgg_master_data, time = "OS_Time_Months", event = "OS_Status", 
                         variables = "CBI", minprop = 0.2)
cutoff_val <- res.cut$cutpoint$cutpoint

low_samples = lgg_master_data$SampleID[lgg_master_data$CBI<cutoff_val]
high_samples = lgg_master_data$SampleID[lgg_master_data$CBI>=cutoff_val]

sub_expr <- transcriptome[transcriptome$index %in% target_samples, -1] %>% 
  tibble::column_to_rownames('index')

# 1.2 Convert per-sample expression to rank and binarize (Top 2000 = 1, Else = 0)
get_top2000_binary <- function(mat, n = 2000) {
  message(paste("Input matrix shape:", nrow(mat), "Samples x", ncol(mat), "Genes"))
  # 1. Rank each row (sample)
  # apply(mat, 1, ...) operates on each row
  # Key point: R's apply function auto-transposes results when operating on rows
  # So ranks result will become [Genes x Samples]
  ranks <- apply(mat, 1, rank, ties.method = "first")
  # 2. Get total gene count (now ranks' row count)
  total_genes <- nrow(ranks)
  # 3. Binarize
  # We need Top N, i.e., Rank > (total - N)
  binary_mat <- ifelse(ranks > (total_genes - n), 1, 0)
  # Now binary_mat is [Genes x Samples]
  # Row names are genes, column names are sample IDs
  return(binary_mat)
}
binary_top2000 <- get_top2000_binary(sub_expr, n = 2000)

mat_high <- binary_top2000[, high_samples]
mat_low <- binary_top2000[, low_samples]

count_high <- rowSums(mat_high) # count per gene in High group
count_low <- rowSums(mat_low)   # count per gene in Low group

n_high <- length(high_samples)
n_low <- length(low_samples)

# Build results data frame
diff_freq_df <- data.frame(
  Gene = rownames(binary_top2000),
  Count_High = count_high,
  Count_Low = count_low
) %>%
  # Filter genes that don't appear in either group (speed up computation)
  filter(Count_High > 0 | Count_Low > 0) %>%
  rowwise() %>%
  mutate(
    # Fisher Test
    P_Value = fisher.test(matrix(c(Count_High, n_high - Count_High, 
                                   Count_Low, n_low - Count_Low), nrow=2))$p.value,
    # Odds Ratio (add epsilon to avoid division by zero)
    Odds_Ratio = (Count_High+0.1 / (n_high - Count_High+0.1)) / 
      ((Count_Low + 0.1) / (n_low - Count_Low + 0.1))
  ) %>%
  ungroup() %>%
  mutate(
    Log2OR = log2(Odds_Ratio),
    FDR = p.adjust(P_Value, method = "BH")
  ) %>%
  arrange(desc(Log2OR))

####enrich

# Assumes gene_list is your sorted vector (Log2OR or other metric)
genes_of_interest <- diff_freq_df %>%
  filter(Log2OR > 0.5 & P_Value < 0.05) %>%
  pull(Gene)

print(paste("Selected genes count:", length(genes_of_interest)))

# 2. Run ORA (enricher)
ora_res <- enricher(
  gene = genes_of_interest,
  TERM2GENE = m_h, # Hallmark
  pvalueCutoff = 0.05,
  pAdjustMethod = "BH"
)

dotplot(ora_res, showCategory=15) + 
  ggtitle("ORA: Pathways Enriched in High CBI Samples")


### VALCANO

# ==============================================================================
# 1. Build known explanation blocklist (Blocklist)
# ==============================================================================

# Previously discovered significant pathways
pathways_to_exclude <- c(
  "HALLMARK_E2F_TARGETS", "HALLMARK_G2M_CHECKPOINT", # Baseline proliferation
  "HALLMARK_KRAS_SIGNALING_UP", "HALLMARK_KRAS_SIGNALING_DN","HALLMARK IL6 JAK STAT3 SIGNALING", # Driver signaling
  "HALLMARK_INFLAMMATORY_RESPONSE", "HALLMARK_TNFA_SIGNALING_VIA_NFKB", # Immune/inflammation
  "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION", "HALLMARK_HYPOXIA" # Stroma/resistance
)

# Extract all genes from these pathways
known_genes <- m_h %>% 
  filter(gs_name %in% pathways_to_exclude) %>% 
  pull(gene_symbol) %>% 
  unique()

print(paste("Number of 'Explained' Genes to exclude:", length(known_genes)))

# ==============================================================================
# 2. Filter Novel Markers
# ==============================================================================
# Conditions:
# 1. Statistically significant (FDR < 0.05)
# 2. Large effect size (|Log2OR| > 1)
# 3. Not in known gene set (!Gene %in% known_genes)

novel_df <- diff_freq_df %>%
  filter(FDR < 0.05, abs(Log2OR) > 1) %>% # First select significant ones
  filter(!Gene %in% known_genes) %>%      # Remove known pathway genes
  mutate(Direction = ifelse(Log2OR > 0, "High CBI (Novel)", "Low CBI (Novel)"))

# Extract Top Candidates (take top 8 from each side)
top_novel_high <- novel_df %>% filter(Direction == "High CBI (Novel)") %>% arrange(desc(Log2OR)) %>% head(8)
top_novel_low  <- novel_df %>% filter(Direction == "Low CBI (Novel)") %>% arrange(Log2OR) %>% head(8)

top_novel_genes <- bind_rows(top_novel_high, top_novel_low)

print("Top Novel High-CBI Markers:")
print(top_novel_high$Gene)

# ==============================================================================
# 3. Prepare plotting data
# ==============================================================================
plot_data_novel <- diff_freq_df %>%
  mutate(
    # Define categories
    Category = case_when(
      Gene %in% top_novel_high$Gene ~ "Novel gene (High CBI)",
      Gene %in% top_novel_low$Gene ~ "Novel gene (Low CBI)",
      Gene %in% known_genes ~ "Known Pathways", # Known genes classified as background
      TRUE ~ "NS"
    ),
    # Define alpha (Novel = opaque, others = transparent)
    Alpha = ifelse(grepl("Novel", Category), 1, 0.3),
    # Define size
    Size = ifelse(grepl("Novel", Category), 2, 1)
  )

# ==============================================================================
# 4. Plot
# ==============================================================================
# Custom colors
color_map <- c(
  "Novel gene (High CBI)" = "#DC143C", # Bright red
  "Novel gene (Low CBI)"  = "#4682B4", # Bright blue
  "Known Pathways" = "black", # Gray background
  "NS" = "grey90"
)

max_val <- max(abs(diff_freq_df$Log2OR), na.rm = TRUE)
limit_x <- max_val * 1.1 

# [Plot code removed -> see figures.qmd]
             aes(color = Category, alpha = Alpha, size = Size)) +
  
  # 2. Highlight points (Novel) - draw on top
  geom_point(data = subset(plot_data_novel, grepl("Novel", Category)),
             aes(color = Category, alpha = Alpha, size = Size), shape=19) +
  
  # 3. Labels (Novel only)
  ggrepel::geom_text_repel(data = subset(plot_data_novel, grepl("Novel", Category)),
                  aes(label = Gene),
                  box.padding = 0.6,
                  max.overlaps = Inf,
                  size = 3,
                  min.segment.length = 0) +
  
  scale_color_manual(values = color_map) +
  scale_alpha_identity() +
  scale_size_identity() +
  
  geom_vline(xintercept = 0, linetype = "dashed") +
  
  labs(
    title = "Discovery of novel features beyond pathways",
    subtitle = "Genes enriched in High/Low CBI inputs but NOT in enriched pathways",
    x = "Log2 Odds Ratio (Frequency Bias)",
    y = "-Log10 P-value"
  ) +
  theme_classic() +
  theme(legend.position = "bottom")+
  xlim(c(-limit_x, limit_x))



################# scRNA
scRNA_pred_tissue = read_delim("/Users/hechang/Documents/chenlab/scRNA/tissue/predictions_comb.csv")
scRNA_pred_blood = read_delim("/Users/hechang/Documents/chenlab/scRNA/blood/predictions_comb.csv")

# ==============================================================================
# 0. Define cell annotations and drug list
# ==============================================================================

# Tissue annotations
annotations_tissue <- list(
  '0' = 'CD8_T', '1' = 'Naive_B', '2' = 'TAMs', '3' = 'T_reg',
  '4' = 'NK', '5' = 'Plasma_Cells', '6' = 'Pr_B', '7' = 'pDCs', '8' = 'Mast_Cells'
)

# Blood annotations
annotations_blood <- c(
  "0" = "NK", "1" = "Naive_T", "2" = "CD14+_Monocytes", "3" = "Naive_B",
  "4" = "gdT", "5" = "Pr_Lymphocytes", "6" = "Plasma_Cells", "7" = "Platelets",
  "8" = "pDCs", "9" = "HSPC"
)

# Cytotoxic drug list (Microtubule inhibitor group)

target_drugs <- drug_sets_list$Microtubule_Inhibitors
#target_drugs = cytotoxic_drugs 

# ==============================================================================
# 1. Generic data processing function (Process Data)
# ==============================================================================
process_scRNA_data <- function(pred_df, annotation_list, drug_filter) {
  
  # Convert annotation list to DF
  if (is.list(annotation_list) && !is.atomic(annotation_list)) {
    anno_df <- tibble(Cluster = names(annotation_list), CellType = unlist(annotation_list))
  } else {
    anno_df <- tibble(Cluster = names(annotation_list), CellType = as.character(annotation_list))
  }
  
  # Data cleaning and merging
  data_clean <- pred_df %>%
    separate_wider_delim(SampleID, '-', names=c('Patient','Status','Cluster'), too_many = "merge") %>%
    mutate(Cluster = as.character(Cluster)) %>%
    left_join(anno_df, by = "Cluster") %>%
    filter(DrugName %in% drug_filter) %>%
    
    # Compute CBI (Z-score then average)
    group_by(DrugName) %>%
    mutate(Z_Score = scale(Predicted_IC50)) %>%
    ungroup() %>%
    group_by(Patient, Status, CellType, Cluster) %>%
    summarise(CBI = mean(Z_Score, na.rm = TRUE), .groups = "drop")
  
  return(data_clean)
}

# Process both datasets
cbi_tissue <- process_scRNA_data(scRNA_pred_tissue, annotations_tissue, target_drugs)
cbi_blood  <- process_scRNA_data(scRNA_pred_blood, annotations_blood, target_drugs)


# ==============================================================================
# Plot Function A: Mechanism Validation (Cell Type Boxplot)
# ==============================================================================
plot_mechanism <- function(data, title_suffix) {
  
  # Sorting: sort by median
  data$CellType <- reorder(data$CellType, data$CBI, FUN = median)
  
# [Plot code removed -> see figures.qmd]
  return(p)
}

# ==============================================================================
# Plot Function B: Longitudinal (Pre vs Post)
# ==============================================================================
plot_longitudinal <- function(data, target_cells, title_suffix) {
  
  # Filter matched patients
  paired_patients <- data %>%
    group_by(Patient) %>%
    filter(any(grepl("Pre", Status, ignore.case=T)) & any(grepl("Post", Status, ignore.case=T))) %>%
    pull(Patient) %>% unique()
  
  plot_data <- data %>%
    filter(Patient %in% paired_patients) %>%
    mutate(Group = case_when(
      grepl("Pre", Status, ignore.case=T) ~ "Pre-treatment",
      grepl("Post", Status, ignore.case=T) ~ "Post-treatment",
      TRUE ~ NA_character_
    )) %>%
    filter(!is.na(Group)) %>%
    filter(CellType %in% target_cells)
  
  plot_data$Group <- factor(plot_data$Group, levels = c("Pre-treatment", "Post-treatment"))
  
# [Plot code removed -> see figures.qmd]
  
  return(p)
}

# ==============================================================================
# Plot Function C: Baseline Prediction (Resistant vs Sensitive)
# ==============================================================================
plot_baseline <- function(data, target_cells, title_suffix) {
  
  # Define outcomes
  patient_outcome <- data %>%
    group_by(Patient) %>%
    summarise(
      Has_Prog = any(grepl("Prog", Status, ignore.case=T)),
      Has_Post = any(grepl("Post", Status, ignore.case=T))
    ) %>%
    mutate(Outcome = case_when(
      Has_Prog ~ "Resistant",
      Has_Post ~ "Sensitive",
      TRUE ~ "Unknown"
    ))
  
  plot_data <- data %>%
    filter(grepl("Pre", Status, ignore.case=T)) %>% 
    left_join(patient_outcome, by = "Patient") %>%
    filter(Outcome != "Unknown") %>%
    filter(CellType %in% target_cells)
  
# [Plot code removed -> see figures.qmd]
  
  return(p)
}

# ==============================================================================
# SET 1: TISSUE Analysis
# ==============================================================================
# Select cells of interest in Tissue (e.g.: Pr_B, CD8_T, NK, Plasma)
cells_tissue <- c("TAMs", "Mast_Cells")
#cells_tissue = annotations_tissue %>% unlist

p1_tissue <- plot_mechanism(cbi_tissue, "Tissue (TME)")
p2_tissue <- plot_longitudinal(cbi_tissue, cells_tissue, "Tissue (TME)")
p3_tissue <- plot_baseline(cbi_tissue, cells_tissue, "Tissue (TME)")

#print(p1_tissue)
# [Plot code removed -> see figures.qmd]
#print(p3_tissue)

# ==============================================================================
# SET 2: BLOOD Analysis
# ==============================================================================
# Select cells of interest in Blood (e.g.: Proliferating, NK, HSPC, Platelets)
cells_blood <- c("HSPC", 'Naive_B',"pDCs")
#cells_blood = annotations_blood %>% unlist()

p1_blood <- plot_mechanism(cbi_blood, "Blood (PBMC)")
p2_blood <- plot_longitudinal(cbi_blood, cells_blood, "Blood (PBMC)")
p3_blood <- plot_baseline(cbi_blood, cells_blood, "Blood (PBMC)")

# [Plot code removed -> see figures.qmd]
 
p2_tissue/p3_blood

tissue_exp = read_delim("/Users/hechang/Documents/chenlab/scRNA/tissue/Tissue_Pseudo_Bulk_by_Cluster_Corrected.csv")

# ==============================================================================
# 0. Preparation: define M2 gene set
# ==============================================================================
# Classic M2 macrophage / immunosuppression-related genes
m2_genes <- c("CD163", "MRC1", "MS4A4A", "STAB1", "TGFB1", "IL10", "FN1", "VSIG4", "MSR1")

# ==============================================================================
# 1. Process expression matrix (tissue_exp)
# ==============================================================================
# Assume tissue_exp has been read
# Format: rows=genes, cols=SampleID (e.g., P002-Post-0)

# 1.1 Transpose matrix (become rows=samples, cols=genes)
# First turn the first column into row names
exp_mat <- tissue_exp %>%
  tibble::column_to_rownames("...1") %>% # Your first column is named ...1
  as.matrix() %>%
  t() %>% 
  as.data.frame()

# 1.2 Filter TAMs (Cluster 2)
# Based on column name P002-Post-0, the last digit is Cluster ID
# Keep only rows ending with "-2" (assuming Cluster 2 = TAMs)
tam_exp <- exp_mat %>%
  tibble::rownames_to_column("Full_ID") %>%
  filter(grepl("-2$", Full_ID)) %>% # Filter Cluster 2
  tibble::column_to_rownames("Full_ID")

# 1.3 Compute M2 Score (Z-score averaging method)
# Ensure genes exist in the matrix
valid_m2_genes <- intersect(m2_genes, colnames(tam_exp))
print(paste("Used M2 Genes:", paste(valid_m2_genes, collapse=", ")))

if(length(valid_m2_genes) > 0) {
  # Z-score normalize expression matrix (by column/gene)
  tam_scaled <- scale(tam_exp)
  
  # Compute mean Z-score of M2 genes
  tam_scores <- data.frame(
    SampleID_Raw = rownames(tam_exp),
    M2_Score = rowMeans(tam_scaled[, valid_m2_genes], na.rm = TRUE)
  )
} else {
  stop("No M2 genes found in expression matrix!")
}

# ==============================================================================
# 2. Merge CBI data
# ==============================================================================
# Assume scRNA_cbi is your previously computed CBI result
# We need to build a matching ID key

# scRNA_cbi has Patient, Status, Cluster
# tissue_exp column name format: Patient-Status-Cluster (e.g., P002-Post-0)

tam_cbi_merged <- cbi_tissue %>%
  # 1. Filter TAMs
  filter(CellType == "TAMs") %>%
  # 2. Build ID consistent with expression matrix
  mutate(SampleID_Raw = paste(Patient, Status, Cluster, sep = "-")) %>%
  # 3. Merge M2 Score
  inner_join(tam_scores, by = "SampleID_Raw")

print(head(tam_cbi_merged))

model <- lm(M2_Score ~ CBI, data = tam_cbi_merged)

# 2. Compute Cook's Distance
cooksd <- cooks.distance(model)

# 3. Define outlier threshold (common standard: 4 / sample size)
n <- nrow(tam_cbi_merged)
threshold <- 4 / n

# 4. Flag and filter outliers
# Keep points where Cook's D < threshold
clean_data <- tam_cbi_merged %>%
  mutate(Cooks_D = cooksd) %>%
  filter(Cooks_D < threshold) # Auto-exclude high-influence points

# Check how many points were removed
n_removed <- n - nrow(clean_data)
print(paste("Auto-removed", n_removed, "outlier(s)"))

# 1. Grouping
# Clean_data here is the outlier-removed data, or use the raw tam_cbi_merged
plot_data_box <- clean_data %>%
  mutate(CBI_Group = ifelse(CBI > median(CBI, na.rm = TRUE), "High Score", "Low Score"))

# Set order
plot_data_box$CBI_Group <- factor(plot_data_box$CBI_Group, levels = c("Low Score", "High Score"))

# [Plot code removed -> see figures.qmd]
  # 1. Add violin plot (show data density distribution)
  geom_violin(
    trim = FALSE, # Set to FALSE to extend to data min/max
    alpha = 0.6,
    scale = "width" # Let violin width reflect sample size
  ) +
  
  # 2. Add simplified boxplot inside violin (optional, for median and quartiles)
  geom_boxplot(
    width = 0.15, # Reduce boxplot width
    outlier.shape = NA,
    fill = "white", # Use white fill for the boxplot
    alpha = 0.5 
  ) +
  
  # 3. Add raw data jitter
  geom_jitter(width = 0.1, size = 2, alpha = 0.5) +
  
  # 4. Statistical test
  stat_compare_means(method = "wilcox.test",label.x = 1.3) + # Adjust p-value label vertical position
  
  # 5. Colors and labels (unchanged)
  scale_fill_manual(values = c("Low Score" = "#4682B4", "High Score" = "#DC143C")) +
  
  labs(
    title = "High Score Indicates Immunosuppression (Violin Plot)",
    x = "Predicted Group",
    y = "M2 Signature Score"
  ) +
  
  # 6. Theme (unchanged)
  theme_classic() +
  theme(legend.position = "none")



# ==============================================================================
# Panel D: Targeting Myeloid Checkpoints 
# ==============================================================================

# Assume you previously extracted tam_exp (TAMs expression matrix)
target_genes <- c("CD274","LILRB4") # or "CD274"
targeting_plots = list()

for (target_gene in target_genes){
  tam_checkpoints <- data.frame(
    SampleID_Raw = rownames(tam_exp),
    Expr = scale(tam_exp[, target_gene]) # Z-score
  )
  
  # 2. Merge CBI (TAMs level)
  plot_data_target <- cbi_tissue %>%
    filter(CellType == "TAMs") %>%
    mutate(SampleID_Raw = paste(Patient, Status, Cluster, sep = "-")) %>%
    inner_join(tam_checkpoints, by = "SampleID_Raw")
  
  model <- lm(Expr ~ CBI, data =plot_data_target)
  
  # 2. Compute Cook's Distance
  cooksd <- cooks.distance(model)
  
  # 3. Define outlier threshold (common standard: 4 / sample size)
  n <- nrow(plot_data_target)
  threshold <- 4 / n
  
  # 4. Flag and filter outliers
  # Keep points where Cook's D < threshold
  clean_data <- plot_data_target %>%
    mutate(Cooks_D = cooksd) %>%
    filter(Cooks_D < threshold) # Auto-exclude high-influence points
  
  # Check how many points were removed
  n_removed <- n - nrow(clean_data)
  print(paste("Auto-removed", n_removed, "outlier(s)"))
  
  
  # 3. Plot
# [Plot code removed -> see figures.qmd]
  
  targeting_plots[[target_gene]] = p_d_new
}
# [Plot code removed -> see figures.qmd]
# combined_plot <- (targeting_plots$CD274 + targeting_plots$LILRB4) + plot_annotation(...)
# print(combined_plot)

# =============================================================================
# Save workspace for figures.qmd
# =============================================================================
dir.create("results", showWarnings = FALSE)
save.image(file = "results/analysis_workspace.RData")
message("Workspace saved to results/analysis_workspace.RData")
