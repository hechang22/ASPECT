# =============================================================================
# Script 6: Downstream Analysis Pipeline
# =============================================================================
# This script contains functions for comprehensive downstream analysis of
# drug sensitivity predictions, including:
# - Model comparison (AUC and Recall)
# - Neighbor lineage analysis
# - Tumor purity correlation analysis
# - Molecular subtype analysis
#
# Usage:
#   source("6_analysis_pipeline.R")
#   run_model_comparison()
#   run_neighbor_analysis("Vinblastine")
#   run_purity_analysis("Vinblastine")
#   run_subtype_analysis("Vinblastine")
# =============================================================================

library(tidyverse)
library(janitor)
library(dplyr)
library(tidyr)
library(tibble)
library(ggplot2)
library(ggpubr)

# =============================================================================
# Section 1: Model Comparison Functions
# =============================================================================

#' Compare Multiple Models
#'
#' @param file_list Named list of validation summary file paths
#' @param output_dir Directory to save plots
#' @return Combined data frame of all model results
compare_models <- function(
    file_list = list(
      "Model_A" = "./validation_results/model_a/validation_summary.csv",
      "Model_B" = "./validation_results/model_b/validation_summary.csv"
    ),
    output_dir = "./figures"
) {
  # Read and merge all CSV files
  all_data <- map_dfr(file_list, ~ read_csv(.x) %>% clean_names(), .id = "model_name")
  
  # Get unique drugs
  drugs_to_compare <- all_data$drug_name %>% unique()
  
  plot_data <- all_data %>%
    filter(drug_name %in% drugs_to_compare) %>%
    mutate(model_name = factor(model_name, levels = names(file_list)))
  
  # Reference model for sorting
  REFERENCE_MODEL <- names(file_list)[1]
  
  # Sort by AUC
  drug_order_auc <- plot_data %>%
    filter(model_name == REFERENCE_MODEL) %>%
    arrange(desc(auc)) %>%
    pull(drug_name)
  
  plot_data_sorted_auc <- plot_data %>%
    mutate(drug_name = factor(drug_name, levels = drug_order_auc))
  
  # Sort by Recall
  drug_order_recall <- plot_data %>%
    filter(model_name == REFERENCE_MODEL) %>%
    arrange(desc(recall_top_n)) %>%
    pull(drug_name)
  
  plot_data_sorted_recall <- plot_data %>%
    mutate(drug_name = factor(drug_name, levels = drug_order_recall))
  
  # Create output directory
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Plot AUC comparison
  auc_plot <- ggplot(plot_data_sorted_auc, aes(x = drug_name, y = auc, fill = model_name)) +
    geom_col(position = position_dodge(width = 0.9)) +
    labs(
      title = "Comparison of Drug Predictive AUC across Models",
      subtitle = "Grouped by Drug",
      x = "Drug Name",
      y = "AUC",
      fill = "Model"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  print(auc_plot)
  ggsave(file.path(output_dir, 'auc_comparison.png'), auc_plot, width = 12, height = 6)
  
  # Plot Recall comparison
  recall_plot <- ggplot(plot_data_sorted_recall, aes(x = drug_name, y = recall_top_n, fill = model_name)) +
    geom_col(position = position_dodge(width = 0.9)) +
    labs(
      title = "Comparison of Drug Recall @ Top-N across Models",
      subtitle = "Grouped by Drug",
      x = "Drug Name",
      y = "Recall @ Top-N",
      fill = "Model"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  print(recall_plot)
  ggsave(file.path(output_dir, 'recall_comparison.png'), recall_plot, width = 12, height = 6)
  
  return(all_data)
}


# =============================================================================
# Section 2: Neighbor Lineage Analysis Functions
# =============================================================================

#' Analyze Neighbor Lineage Enrichment
#'
#' @param pred_file Path to predictions CSV
#' @param ground_truth_file Path to ground truth matrix CSV
#' @param ccle_meta_file Path to CCLE metadata CSV
#' @param tcga_md_file Path to TCGA metadata CSV
#' @param drug_name Drug name to analyze
#' @param top_n Number of top predictions to analyze
#' @param output_dir Directory to save outputs
#' @return Analysis results list
analyze_neighbor_lineage <- function(
    pred_file = "./results/predictions.csv",
    ground_truth_file = "./validation_results/ground_truth_sensitivity_matrix.csv",
    ccle_meta_file = "./data/metadata/Model.csv",
    tcga_md_file = "./data/metadata/clinical_TumorCompendium_v11_PolyA.tsv",
    drug_name = "Vinblastine",
    top_n = 600,
    output_dir = "./figures"
) {
  # Load data
  pred_full <- read_delim(pred_file)
  ground_truth <- read_delim(ground_truth_file)
  ground_truth <- ground_truth %>% 
    pivot_longer(-DrugName, names_to = 'SampleID', values_to = "Truth")
  pred_full <- pred_full %>% left_join(ground_truth)
  
  ccle_meta <- read_delim(ccle_meta_file)
  tcga_md <- read_delim(tcga_md_file)
  
  # Get top predictions
  pred_top <- pred_full %>%
    filter(DrugName == drug_name) %>%
    arrange(-Predicted_IC50) %>%
    slice_head(n = top_n)
  
  # Extract neighbors for on-label samples
  sub_neighbors <- pred_top %>%
    filter(Truth == 1) %>%
    separate_rows(KNN_Neighbors, sep = ";") %>%
    filter(KNN_Neighbors != "" & !is.na(KNN_Neighbors)) %>%
    count(KNN_Neighbors, sort = TRUE, name = "Count")
  
  # Annotate with metadata
  neighbor_annotated <- sub_neighbors %>%
    left_join(ccle_meta, by = c("KNN_Neighbors" = "ModelID"))
  
  analysis_population <- neighbor_annotated %>%
    uncount(Count)
  
  background_population <- ccle_meta
  
  # Calculate observed counts
  obs_counts <- analysis_population %>%
    filter(!is.na(PrimaryOrMetastasis)) %>%
    count(PrimaryOrMetastasis) %>%
    mutate(Group = "Selected_Neighbors")
  
  # Calculate background counts
  bg_counts <- background_population %>%
    filter(!is.na(PrimaryOrMetastasis)) %>%
    count(PrimaryOrMetastasis) %>%
    mutate(Group = "Background_CCLE")
  
  # Create contingency table
  contingency_data <- bind_rows(obs_counts, bg_counts) %>%
    pivot_wider(names_from = PrimaryOrMetastasis, values_from = n, values_fill = 0) %>%
    column_to_rownames("Group")
  
  # Chi-square test
  chisq_result <- chisq.test(contingency_data)
  
  print(paste("Chi-square test for", drug_name))
  print(chisq_result)
  
  # Plot proportions
  plot_data <- bind_rows(obs_counts, bg_counts) %>%
    group_by(Group) %>%
    mutate(Prop = n / sum(n)) %>%
    rename(Status = PrimaryOrMetastasis)
  
  p <- ggplot(plot_data, aes(x = Group, y = Prop, fill = Status)) +
    geom_col(position = "fill") +
    labs(title = paste("Proportion of Metastatic Samples -", drug_name), y = "Proportion") +
    theme_minimal()
  
  print(p)
  
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  ggsave(file.path(output_dir, paste0('metastasis_proportion_', drug_name, '.png')), p)
  
  # Lineage enrichment analysis
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
  
  print("Lineage Enrichment:")
  print(enrichment)
  
  return(list(
    chi_square = chisq_result,
    enrichment = enrichment,
    contingency = contingency_data
  ))
}


#' Cross-Cancer Lineage Analysis
#'
#' Creates a bubble plot showing lineage enrichment across cancer types
analyze_cross_cancer_lineage <- function(
    pred_file = "./results/predictions.csv",
    ground_truth_file = "./validation_results/ground_truth_sensitivity_matrix.csv",
    ccle_meta_file = "./data/metadata/Model.csv",
    tcga_md_file = "./data/metadata/clinical_TumorCompendium_v11_PolyA.tsv",
    drug_name = "Vinblastine",
    output_dir = "./figures"
) {
  # Load data
  pred_full <- read_delim(pred_file)
  ground_truth <- read_delim(ground_truth_file)
  ground_truth <- ground_truth %>% 
    pivot_longer(-DrugName, names_to = 'SampleID', values_to = "Truth")
  pred_full <- pred_full %>% left_join(ground_truth)
  
  ccle_meta <- read_delim(ccle_meta_file)
  tcga_md <- read_delim(tcga_md_file)
  
  # Prepare full data
  full_data <- pred_full %>%
    filter(Truth == 1) %>%
    separate_rows(KNN_Neighbors, sep = ";") %>%
    filter(KNN_Neighbors != "" & !is.na(KNN_Neighbors)) %>%
    left_join(tcga_md, by = c("SampleID" = "th_dataset_id")) %>%
    left_join(ccle_meta, by = c('KNN_Neighbors' = 'ModelID'))
  
  # Background stats
  bg_stats <- ccle_meta %>%
    count(OncotreeLineage) %>%
    mutate(Freq_BG = n / sum(n)) %>%
    select(OncotreeLineage, Freq_BG)
  
  # Top cancers
  target_cancers <- table(tcga_md$disease) %>% 
    sort(decreasing = TRUE) %>% 
    head(15) %>% 
    names()
  
  # Analyze each cancer
  plot_data_list <- list()
  
  for (cancer in target_cancers) {
    cancer_neighbors <- full_data %>%
      filter(disease == cancer)
    
    total_neighbors <- nrow(cancer_neighbors)
    
    obs_stats <- cancer_neighbors %>%
      count(OncotreeLineage) %>%
      mutate(Freq_Selected = n / sum(n)) %>%
      rename(n_Obs = n)
    
    enrichment_res <- obs_stats %>%
      left_join(bg_stats, by = "OncotreeLineage") %>%
      mutate(
        Fold_Change = Freq_Selected / Freq_BG,
        CancerType = cancer
      ) %>%
      filter(n_Obs > 0)
    
    plot_data_list[[cancer]] <- enrichment_res
  }
  
  final_plot_data <- bind_rows(plot_data_list)
  
  # Highlight specific lineages
  highlight_lineages <- c("Esophagus/Stomach", "Breast", "Pancreas", "Ovary/Fallopian Tube")
  
  viz_data <- final_plot_data %>%
    filter(OncotreeLineage %in% highlight_lineages | Fold_Change > 2.0) %>%
    mutate(OncotreeLineage = fct_relevel(OncotreeLineage, highlight_lineages)) %>%
    mutate(FC_Plot = pmin(Fold_Change, 6))
  
  p <- ggplot(viz_data, aes(x = CancerType, y = OncotreeLineage)) +
    geom_point(aes(size = FC_Plot, color = FC_Plot), alpha = 0.9) +
    scale_color_gradientn(
      colors = c("grey90", "#FFB6C1", "#DC143C", "#8B0000"),
      values = c(0, 0.2, 0.6, 1),
      name = "Fold Change"
    ) +
    scale_size_continuous(range = c(2, 8), name = "Enrichment\nMagnitude") +
    labs(
      title = "Lineage confounding analysis",
      subtitle = "CCLE lineages enriched in neighbors of patients across TCGA cancers",
      x = "TCGA Cancer Type",
      y = "CCLE Neighbor Lineage"
    ) +
    theme_bw() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
      axis.text.y = element_text(face = "bold", size = 10),
      panel.grid.major = element_line(color = "grey95"),
      legend.position = "right"
    ) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0.5, ymax = 4.5,
             alpha = 0, color = "blue", linetype = "dashed", size = 1)
  
  print(p)
  
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  ggsave(file.path(output_dir, paste0('lineage_confounding_', drug_name, '.png')), 
         p, width = 14, height = 8)
  
  return(final_plot_data)
}


# =============================================================================
# Section 3: Tumor Purity Analysis Functions
# =============================================================================

#' Analyze Tumor Purity Correlation
#'
#' @param pred_file Path to predictions CSV
#' @param est_score_file Path to ESTIMATE scores
#' @param tcga_md_file Path to TCGA metadata
#' @param drug_name Drug name to analyze
#' @param output_dir Directory to save outputs
analyze_tumor_purity <- function(
    pred_file = "./results/predictions.csv",
    est_score_file = "./data/TCGA_estimate_score.gct",
    tcga_md_file = "./data/metadata/clinical_TumorCompendium_v11_PolyA.tsv",
    drug_name = "Vinblastine",
    output_dir = "./figures"
) {
  # Load data
  pred_full <- read_delim(pred_file)
  est_score <- read_delim(est_score_file)
  tcga_md <- read_delim(tcga_md_file)
  
  # Clean ESTIMATE scores
  est_clean <- est_score %>%
    select(-Description) %>%
    pivot_longer(cols = -NAME, names_to = "SampleID", values_to = "Score") %>%
    pivot_wider(names_from = NAME, values_from = Score)
  
  # Merge data
  cor_data <- est_clean %>%
    inner_join(pred_full %>% filter(DrugName == drug_name), by = "SampleID")
  
  # Create purity groups
  plot_data <- cor_data %>%
    mutate(Purity_Group = cut(TumorPurity,
                              breaks = c(0, 0.6, 0.8, 1.0),
                              labels = c("Low", "Medium", "High")))
  
  # Boxplot by purity group
  p <- ggplot(plot_data, aes(x = Purity_Group, y = Predicted_IC50, fill = Purity_Group)) +
    geom_boxplot(outlier.alpha = 0.3) +
    stat_compare_means(comparisons = list(c("Low", "High")), method = "wilcox.test") +
    scale_fill_brewer(palette = "Reds") +
    labs(
      title = paste("By Tumor purity -", drug_name),
      x = "ESTIMATE Tumor Purity Group",
      y = "Predicted Score"
    ) +
    theme_classic()
  
  print(p)
  
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  ggsave(file.path(output_dir, paste0('tumor_purity_', drug_name, '.png')), p)
  
  # Cancer-specific analysis
  merged_data <- cor_data %>%
    left_join(tcga_md %>% select(th_dataset_id, CancerName = disease), 
              by = c("SampleID" = "th_dataset_id"))
  
  cancer_stats <- merged_data %>%
    group_by(CancerName) %>%
    filter(n() > 30) %>%
    summarise(
      N = n(),
      Corr_R = cor(TumorPurity, Predicted_IC50, method = "pearson", use = "complete.obs"),
      P_Value = tryCatch(
        cor.test(TumorPurity, Predicted_IC50, method = "pearson")$p.value,
        error = function(e) 1
      )
    ) %>%
    ungroup()
  
  # Select top cancers with positive correlation
  target_cancers_df <- cancer_stats %>%
    filter(Corr_R > 0) %>%
    arrange(P_Value) %>%
    slice_head(n = 12)
  
  target_cancers <- target_cancers_df$CancerName
  
  # Plot subset
  plot_data_subset <- merged_data %>%
    filter(CancerName %in% target_cancers) %>%
    mutate(CancerName = factor(CancerName, levels = target_cancers))
  
  p2 <- ggplot(plot_data_subset, aes(x = TumorPurity, y = Predicted_IC50)) +
    geom_point(alpha = 0.3, color = "royalblue", size = 1) +
    geom_smooth(method = "lm", color = "red", fill = "pink", alpha = 0.5) +
    stat_cor(method = "pearson", label.x.npc = "left", label.y.npc = "top", size = 3) +
    facet_wrap(~ CancerName, scales = "free") +
    labs(
      title = paste("Tumor Purity vs. Scores -", drug_name),
      x = "ESTIMATE Tumor Purity",
      y = "Predicted Score"
    ) +
    theme_bw() +
    theme(
      strip.background = element_rect(fill = "grey90"),
      strip.text = element_text(face = "bold", size = 8),
      axis.text = element_text(size = 8)
    )
  
  print(p2)
  ggsave(file.path(output_dir, paste0('tumor_purity_by_cancer_', drug_name, '.png')), 
         p2, width = 12, height = 10)
  
  return(list(
    cancer_stats = cancer_stats,
    target_cancers = target_cancers_df
  ))
}


# =============================================================================
# Section 4: Molecular Subtype Analysis Functions
# =============================================================================

#' Analyze Molecular Subtypes
#'
#' @param pred_file Path to predictions CSV
#' @param drug_name Drug name to analyze
#' @param cancer_type Cancer type (e.g., "BRCA", "OV")
#' @param output_dir Directory to save outputs
analyze_molecular_subtypes <- function(
    pred_file = "./results/predictions.csv",
    drug_name = "Vinblastine",
    cancer_type = "BRCA",
    output_dir = "./figures"
) {
  # Load TCGA subtypes
  library(TCGAbiolinks)
  subtypes_data <- PanCancerAtlas_subtypes()
  
  # Clean subtypes
  clean_subtypes <- subtypes_data %>%
    select(
      SampleID = pan.samplesID,
      CancerCode = cancer.type,
      Subtype = Subtype_Selected
    ) %>%
    mutate(SampleID = substr(SampleID, 1, 15)) %>%
    mutate(SampleID = gsub("\\.", "-", SampleID))
  
  # Load predictions
  pred_full <- read_delim(pred_file)
  
  my_pred <- pred_full %>%
    filter(DrugName == drug_name) %>%
    mutate(SampleID = substr(SampleID, 1, 15)) %>%
    mutate(SampleID = gsub("\\.", "-", SampleID))
  
  # Merge
  merged_analysis <- my_pred %>%
    inner_join(clean_subtypes, by = "SampleID")
  
  if (cancer_type == "BRCA") {
    # BRCA subtype analysis
    brca_data <- merged_analysis %>%
      filter(CancerCode == "BRCA") %>%
      filter(Subtype != 'BRCA.Normal')
    
    brca_data$Subtype <- factor(brca_data$Subtype,
                                levels = c("BRCA.LumA", "BRCA.LumB", "BRCA.Her2", "BRCA.Basal"))
    
    my_comparisons <- list(c("BRCA.LumA", "BRCA.Basal"), c("BRCA.LumA", "BRCA.LumB"))
    
    p <- ggplot(brca_data, aes(x = Subtype, y = Predicted_IC50, fill = Subtype)) +
      geom_boxplot(outlier.shape = NA, alpha = 0.7) +
      geom_jitter(width = 0.2, alpha = 0.2, size = 1) +
      scale_fill_brewer(palette = "RdBu", direction = -1) +
      stat_compare_means(comparisons = my_comparisons,
                         method = "wilcox.test",
                         label = "p.signif",
                         size = 5) +
      stat_compare_means(label.y = max(brca_data$Predicted_IC50) * 1.1) +
      labs(
        title = paste("Validation by Molecular Subtypes: BRCA -", drug_name),
        subtitle = "Higher Score = Higher Proliferation/Sensitivity",
        x = "PAM50 Subtype",
        y = "Predicted Score (IC50)"
      ) +
      theme_classic() +
      theme(
        legend.position = "none",
        axis.text.x = element_text(face = "bold", size = 11),
        plot.title = element_text(face = "bold")
      )
    
    print(p)
    
  } else if (cancer_type == "OV") {
    # Ovarian cancer subtype analysis
    ov_data <- merged_analysis %>%
      filter(grepl("OVCA", Subtype))
    
    ov_data$Subtype <- factor(ov_data$Subtype,
                              levels = c("OVCA.Mesenchymal", "OVCA.Differentiated",
                                         "OVCA.Immunoreactive", "OVCA.Proliferative"))
    
    my_comparisons_ov <- list(c("OVCA.Mesenchymal", "OVCA.Proliferative"))
    
    p <- ggplot(ov_data, aes(x = Subtype, y = Predicted_IC50, fill = Subtype)) +
      geom_boxplot(outlier.shape = NA, alpha = 0.7) +
      geom_jitter(width = 0.2, alpha = 0.2, size = 1) +
      scale_fill_manual(values = c("OVCA.Mesenchymal" = "#4682B4",
                                   "OVCA.Differentiated" = "grey",
                                   "OVCA.Immunoreactive" = "grey",
                                   "OVCA.Proliferative" = "#DC143C")) +
      stat_compare_means(comparisons = my_comparisons_ov, method = "wilcox.test", label = "p.signif") +
      stat_compare_means(label.y = max(ov_data$Predicted_IC50) * 1.05) +
      labs(
        title = paste("Validation in Ovarian Cancer -", drug_name),
        subtitle = "Proliferative Subtype -> Highest IC50",
        x = "Subtype",
        y = "Predicted Score"
      ) +
      theme_classic() +
      theme(
        legend.position = "none",
        axis.text.x = element_text(face = "bold", size = 10, angle = 45, hjust = 1),
        plot.title = element_text(face = "bold")
      )
    
    print(p)
  }
  
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  ggsave(file.path(output_dir, paste0('subtype_analysis_', cancer_type, '_', drug_name, '.png')), 
         p, width = 8, height = 6)
  
  return(merged_analysis)
}


# =============================================================================
# Section 5: Main Execution Functions
# =============================================================================

#' Run Complete Analysis Pipeline
#'
#' @param config List of configuration parameters
run_complete_analysis <- function(config = list()) {
  # Default configuration
  default_config <- list(
    pred_file = "./results/predictions.csv",
    ground_truth_file = "./validation_results/ground_truth_sensitivity_matrix.csv",
    ccle_meta_file = "./data/metadata/Model.csv",
    tcga_md_file = "./data/metadata/clinical_TumorCompendium_v11_PolyA.tsv",
    est_score_file = "./data/TCGA_estimate_score.gct",
    drug_name = "Vinblastine",
    output_dir = "./figures"
  )
  
  # Merge with user config
  cfg <- modifyList(default_config, config)
  
  # Run analyses
  cat("Running neighbor lineage analysis...\n")
  neighbor_results <- analyze_neighbor_lineage(
    pred_file = cfg$pred_file,
    ground_truth_file = cfg$ground_truth_file,
    ccle_meta_file = cfg$ccle_meta_file,
    tcga_md_file = cfg$tcga_md_file,
    drug_name = cfg$drug_name,
    output_dir = cfg$output_dir
  )
  
  cat("Running cross-cancer lineage analysis...\n")
  cross_cancer_results <- analyze_cross_cancer_lineage(
    pred_file = cfg$pred_file,
    ground_truth_file = cfg$ground_truth_file,
    ccle_meta_file = cfg$ccle_meta_file,
    tcga_md_file = cfg$tcga_md_file,
    drug_name = cfg$drug_name,
    output_dir = cfg$output_dir
  )
  
  cat("Running tumor purity analysis...\n")
  purity_results <- analyze_tumor_purity(
    pred_file = cfg$pred_file,
    est_score_file = cfg$est_score_file,
    tcga_md_file = cfg$tcga_md_file,
    drug_name = cfg$drug_name,
    output_dir = cfg$output_dir
  )
  
  cat("Running molecular subtype analysis...\n")
  subtype_results <- analyze_molecular_subtypes(
    pred_file = cfg$pred_file,
    drug_name = cfg$drug_name,
    cancer_type = "BRCA",
    output_dir = cfg$output_dir
  )
  
  cat("Analysis complete!\n")
  
  return(list(
    neighbor = neighbor_results,
    cross_cancer = cross_cancer_results,
    purity = purity_results,
    subtype = subtype_results
  ))
}


# =============================================================================
# Example Usage (commented out)
# =============================================================================

# Example 1: Model comparison
# compare_models(
#   file_list = list(
#     "CellHit" = "./validation/cellhit/validation_summary.csv",
#     "C2S" = "./validation/c2s/validation_summary.csv"
#   ),
#   output_dir = "./figures"
# )

# Example 2: Complete analysis for a drug
# results <- run_complete_analysis(list(
#   drug_name = "Vinblastine",
#   output_dir = "./figures/vinblastine"
# ))

# Example 3: Individual analyses
# analyze_neighbor_lineage(drug_name = "Irinotecan")
# analyze_tumor_purity(drug_name = "Sorafenib")
# analyze_molecular_subtypes(drug_name = "Niraparib", cancer_type = "OV")
