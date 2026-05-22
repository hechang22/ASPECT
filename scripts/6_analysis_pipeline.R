# =============================================================================
# ASPECT Downstream Analysis Pipeline
# =============================================================================
# This script contains the complete downstream analysis for drug sensitivity
# predictions. Run sections sequentially after completing steps 1-5.
#
# Prerequisites: predictions.csv, validation outputs, ESTIMATE scores,
#                TCGA clinical data, TCGA subtypes (TCGAbiolinks).
# =============================================================================

# ---- Load all required libraries ----
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
library(ranger)
library(survival)
library(survminer)
library(forestmodel)
library(broom)
library(psych)
library(stringr)




# =============================================================================
# Input File Configuration
# =============================================================================
file_list <- list(
  "CellHit" = "/Users/hechang/Documents/chenlab/CellHit_onlyCPU/scripts/feat_C2s/validation_per_drug/all_drugs_validation_summary.csv",
  "2k+2k" = "/Users/hechang/Documents/chenlab/CellHit_onlyCPU/scripts/feat_C2s/validation_per_drug_2k_reversed/all_drugs_validation_summary_singlefile.csv",
  "Comb" = "/Users/hechang/Documents/chenlab/CellHit_onlyCPU/scripts/feat_C2s/validation_per_drug_cclev2_reversed/all_drugs_validation_summary_singlefile.csv"
)

# -------------------------------------------------------------------------
# 步骤 1: 读取并合并所有 CSV 文件
# -------------------------------------------------------------------------
# 我们使用 purrr::map_dfr 来循环读取列表中的每个文件，
# 并将它们合并到一个数据框中。
# .id = "model_name" 会自动创建一个新列 (名为 "model_name")，
# 它的值来自 file_list 中的 "名字" (例如 "Model_A", "Model_B")。


# =============================================================================
# Section 1: Model Comparison (AUC & Recall)
# =============================================================================
all_data <- map_dfr(file_list, ~ read_csv(.x) %>% clean_names(), .id = "model_name")

# clean_names() 会自动将 "Recall @ Top-N" 转换为 "recall_at_top_n"，
# 将 "AUC" 转换为 "auc"，这样在代码中引用它们会更方便。

# 检查合并后的数据
print(all_data)

# -------------------------------------------------------------------------
# 步骤 4: (可选，但推荐) 筛选您感兴趣的药物
# -------------------------------------------------------------------------
# 如果您的文件包含成百上千种药物，直接绘图会导致图表混乱不堪。
# 通常，您会想筛选几个特定的药物进行比较。

# 从示例数据中选择要比较的药物
#drugs_to_compare <- c("Vinblastine","Irinotecan","Sorafenib","Niraparib")
drugs_to_compare = all_data$drug_name %>%unique()
  
plot_data <- all_data %>%
  filter(drug_name %in% drugs_to_compare) %>% 
  mutate(model_name= factor(model_name, levels = c('CellHit','2k+2k','Comb')))

# *** 在这里选择您想用作排序基准的模型 ***
REFERENCE_MODEL <- "CellHit" # 您可以改成 "Model_B" 或 file_list 中的任何名字

# --- 为 AUC 图表排序 ---

# 1. 筛选出基准模型的数据
# 2. 按照 `auc` 列进行降序 (desc) 排列
# 3. 提取 (pull) `drug_name` 列作为一个向量。这就是我们想要的顺序。
drug_order_auc <- plot_data %>%
  filter(model_name == REFERENCE_MODEL) %>%
  arrange(desc(auc)) %>%
  pull(drug_name)


# 4. 使用这个顺序来 "诱变" (mutate) plot_data 中的 drug_name 列
# 我们将其转换为一个因子 (factor)，并明确指定其水平 (levels)
plot_data_sorted_auc <- plot_data %>%
  mutate(drug_name = factor(drug_name, levels = drug_order_auc))


# --- (可选) 为 Recall 图表创建 *不同* 的排序 ---
# 注意：Recall 的排序可能与 AUC 不同，所以我们单独计算
drug_order_recall <- plot_data %>%
  filter(model_name == REFERENCE_MODEL) %>%
  arrange(desc(recall_top_n)) %>%
  pull(drug_name)

plot_data_sorted_recall <- plot_data %>%
  mutate(drug_name = factor(drug_name, levels = drug_order_recall))

# -------------------------------------------------------------------------
# 步骤 5: 绘制分组柱状图 (比较 AUC)
# -------------------------------------------------------------------------
auc_plot <- ggplot(plot_data_sorted_auc, aes(x = drug_name, y = auc, fill = model_name)) +
  geom_col(position = position_dodge(width = 0.9)) +
  labs(
    title = "Comparison of Drug Predictive AUC across Models",
    subtitle = "Grouped by Drug",
    x = "Drug Name",
    y = "AUC",
    fill = "Model"  # 这是图例的标题
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # 当x轴标签过长时旋转

# 显示图表
print(auc_plot)
ggsave('/Users/hechang/Documents/chenlab/fig/auc_plot_reversed.png',auc_plot)
# -------------------------------------------------------------------------
# 步骤 6: 绘制分组柱状图 (比较 Recall)
# -------------------------------------------------------------------------
# 代码几乎完全相同，只需将 `y = auc` 改为 `y = recall_at_top_n`

recall_plot <- ggplot(plot_data_sorted_recall, aes(x = drug_name, y =recall_top_n, fill = model_name)) +
  geom_col(position = position_dodge(width = 0.9)) +
  labs(
    title = "Comparison of Drug Recall @ Top-N across Models",
    subtitle = "Grouped by Drug",
    x = "Drug Name",
    y = "Recall @ Top-N",
    fill = "Model" # 图例标题
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 显示图表
print(recall_plot)
ggsave('/Users/hechang/Documents/chenlab/fig/recall_plot_reversed.png',recall_plot)

######

# =============================================================================
# Section 2: Neighbor Lineage & Metastasis Analysis
# =============================================================================
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

# 提取背景库的分布
bg_counts <- background_population %>%
  filter(!is.na(PrimaryOrMetastasis)) %>%
  count(PrimaryOrMetastasis) %>%
  mutate(Group = "Background_CCLE")

# 合并
contingency_data <- bind_rows(obs_counts, bg_counts) %>%
  pivot_wider(names_from = PrimaryOrMetastasis, values_from = n, values_fill = 0) %>%
  column_to_rownames("Group")

# 2. 执行卡方检验
chisq_result <- chisq.test(contingency_data)

# 3. 输出结果
print(chisq_result)
print(chisq_result$observed) # 观察值
print(chisq_result$expected) # 期望值

plot_data <- bind_rows(obs_counts, bg_counts) %>%
  group_by(Group) %>%
  mutate(Prop = n / sum(n)) %>%
  rename(Status = PrimaryOrMetastasis)

ggplot(plot_data, aes(x = Group, y = Prop, fill =Status)) +
  geom_col(position = "fill") +
  labs(title = "Proportion of Metastatic Samples", y = "Proportion") +
  theme_minimal()


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


# =============================================================================
# Section 3: Cross-Cancer Lineage Enrichment
# =============================================================================
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
  
  # 3.1 提取该癌种下的所有邻居
  cancer_neighbors <- full_data %>%
    filter(disease == cancer)
  
  total_neighbors <- nrow(cancer_neighbors)
  
  # 3.2 计算该癌种选中的邻居谱系频率
  obs_stats <- cancer_neighbors %>%
    count(OncotreeLineage) %>%
    mutate(Freq_Selected = n / sum(n)) %>%
    rename(n_Obs = n)
  
  # 3.3 合并背景并计算 Fold Change
  enrichment_res <- obs_stats %>%
    left_join(bg_stats, by = "OncotreeLineage") %>%
    mutate(
      Fold_Change = Freq_Selected / Freq_BG,
      CancerType = cancer # 标记当前列是哪个癌种
    ) %>%
    # 只保留 FC > 1 的主要富集项，或者是你关注的特定谱系
    filter(n_Obs > 0) # 过滤掉噪音
  
  plot_data_list[[cancer]] <- enrichment_res
}

# 合并所有结果
final_plot_data <- bind_rows(plot_data_list)

highlight_lineages <- c("Esophagus/Stomach", "Breast", "Pancreas", "Ovary/Fallopian Tube")

viz_data <- final_plot_data %>%
  # 1. 过滤：只展示 Fold Change 比较高的，或者属于重点谱系的
  filter(OncotreeLineage %in% highlight_lineages | Fold_Change > 2.0) %>%
  
  # 2. Y轴排序：让最显著的 Stomach 排在最上面
  mutate(OncotreeLineage = fct_relevel(OncotreeLineage, highlight_lineages)) %>%
  mutate(FC_Plot = pmin(Fold_Change, 6))

ggplot(viz_data, aes(x = CancerType, y = OncotreeLineage)) +
  # 画气泡
  geom_point(aes(size = FC_Plot, color = FC_Plot), alpha = 0.9) +
  
  # 颜色设置：使用红炎色系，颜色越深代表富集越强
  scale_color_gradientn(
    colors = c("grey90", "#FFB6C1", "#DC143C", "#8B0000"), 
    values = c(0, 0.2, 0.6, 1),
    name = "Fold Change"
  ) +
  
  # 大小设置
  scale_size_continuous(range = c(2, 8), name = "Enrichment\nMagnitude") +
  
  # 坐标轴与主题
  labs(
    title = "Lineage confounding analysis",
    subtitle = "CCLE lineages enriched in neighbors of patients across TCGA cancers",
    x = "TCGA Cancer Type",
    y = "CCLE Neighbor Lineage"
  ) +
  
  theme_bw() +
  theme(
    # X轴文字旋转，防止重叠
    axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
    # Y轴文字加粗
    axis.text.y = element_text(face = "bold", size = 10),
    # 网格线微调
    panel.grid.major = element_line(color = "grey95"),
    legend.position = "right"
  ) +
  
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0.5, ymax = 4.5, 
           alpha = 0, color = "blue", linetype = "dashed", size = 1)

##### estimate


# =============================================================================
# Section 4: Tumor Purity vs Predicted Scores
# =============================================================================
est_clean <- est_score %>%
  select(-Description) %>% # 去掉 Description 列，只留 NAME 和 样本列
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

ggplot(plot_data, aes(x = Purity_Group, y = Predicted_IC50, fill = Purity_Group)) +
  geom_boxplot(outlier.alpha = 0.3) +
  # 添加 P 值比较 (比较 Low vs High)
  stat_compare_means(comparisons = list(c("Low", "High")), 
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
  # 过滤掉样本量太少的癌种（例如少于30个），否则相关性计算不可靠
  filter(n() > 30) %>% 
  summarise(
    N = n(),
    Corr_R = cor(TumorPurity, Predicted_IC50, method = "pearson", use = "complete.obs"),
    # 使用 tryCatch 防止某些极端情况报错
    P_Value = tryCatch(
      cor.test(TumorPurity, Predicted_IC50, method = "pearson")$p.value,
      error = function(e) 1
    )
  ) %>%
  ungroup()

# 查看一下计算结果
print(head(cancer_stats %>% arrange(P_Value)))

target_cancers_df <- cancer_stats %>%
  # 1. 只看正相关 (支持你的恶性增殖假说)
  filter(Corr_R > 0) %>%
  # 2. 按 P 值从小到大排序 (最显著的排前面)
  arrange(P_Value) %>%
  # 3. 选前 12 个
  slice_head(n = 12)

target_cancers <- target_cancers_df$CancerName

print("Selected Positive Correlation Cancers:")
print(target_cancers_df)

# target_cancers_df <- cancer_stats %>%
#   arrange(P_Value) %>%
#   slice_head(n = 12)
# target_cancers <- target_cancers_df$CancerName


# --- 准备绘图数据 ---
plot_data_subset <- merged_data %>%
  filter(CancerName %in% target_cancers) %>%
  mutate(CancerName = factor(CancerName, levels = target_cancers))

# --- 绘图 ---
ggplot(plot_data_subset, aes(x = TumorPurity, y = Predicted_IC50)) +
  # 1. 散点
  geom_point(alpha = 0.3, color = "royalblue", size = 1) +
  
  # 2. 回归线
  geom_smooth(method = "lm", color = "red", fill = "pink", alpha = 0.5) +
  
  # 3. 统计值标注
  stat_cor(method = "pearson", label.x.npc = "left", label.y.npc = "top", size = 3) +
  
  # 4. 分面 (scales="free" 依然很重要)
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

# =============================================================================
# Section 5: Molecular Subtype Validation
# =============================================================================
subtypes_data <- PanCancerAtlas_subtypes()
head(subtypes_data)

#drugs_to_compare <- c("Vinblastine","Irinotecan","Sorafenib","Niraparib")
DRUG='Vinblastine'

clean_subtypes <- subtypes_data %>%
  select(
    SampleID = pan.samplesID,
    CancerCode = cancer.type,
    Subtype = Subtype_Selected # 或者是 Subtype_mRNA, 视具体癌种而定
  ) %>%
  # 统一 ID 格式：前12-15位字符，并将 . 替换为 -
  mutate(SampleID = substr(SampleID, 1, 15)) %>% 
  mutate(SampleID = gsub("\\.", "-", SampleID))

# 准备你的预测数据 (以 Vinblastine 为例)
my_pred <- pred_full %>%
  filter(DrugName == DRUG) %>% # 替换成你的药物名
  mutate(SampleID = substr(SampleID, 1, 15)) %>% # 确保 ID 长度一致以便匹配
  mutate(SampleID = gsub("\\.", "-", SampleID))

# 合并
merged_analysis <- my_pred %>%
  inner_join(clean_subtypes, by = "SampleID")


brca_data <- merged_analysis %>%
  filter(CancerCode == "BRCA")%>%
  filter(Subtype!='BRCA.Normal')

brca_data$Subtype <- factor(brca_data$Subtype, 
                            levels = c("BRCA.LumA", "BRCA.LumB", "BRCA.Her2", "BRCA.Basal"))

# 定义我们要重点比较的组：LumA vs Basal
my_comparisons <- list(c("BRCA.LumA", "BRCA.Basal"), c("BRCA.LumA", "BRCA.LumB"))

ggplot(brca_data, aes(x = Subtype, y = Predicted_IC50, fill = Subtype)) +
  # 1. 箱线图 + 散点
  geom_boxplot(outlier.shape = NA, alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.2, size = 1) +
  
  # 2. 颜色设置 (红蓝配色常用于区分冷热肿瘤)
  scale_fill_brewer(palette = "RdBu", direction = -1) + # 红色给 Basal (Hot/Aggressive)
  
  # 3. 添加统计显著性 (Wilcoxon test)
  stat_compare_means(comparisons = my_comparisons, 
                     method = "wilcox.test", 
                     label = "p.signif", # 显示星号 (*, **, ***)
                     size = 5) +
  
  # 4. 添加全局 P 值 (Kruskal-Wallis)
  stat_compare_means(label.y = max(brca_data$Predicted_IC50) * 1.1) + 
  
  # 5. 标题与标注
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
  filter(grepl("OVCA", Subtype)) # 提取 OV 相关亚型

# 设定顺序：把我们最想对比的 Mesenchymal 和 Proliferative 放在两端
ov_data$Subtype <- factor(ov_data$Subtype, 
                          levels = c("OVCA.Mesenchymal", "OVCA.Differentiated", 
                                     "OVCA.Immunoreactive", "OVCA.Proliferative"))

# 重点比较 Mesenchymal (低纯度/基质多) vs Proliferative (高纯度/高增殖)
my_comparisons_ov <- list(c("OVCA.Mesenchymal", "OVCA.Proliferative"))

ggplot(ov_data, aes(x = Subtype, y = Predicted_IC50, fill = Subtype)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.2, size = 1) +
  
  # 颜色：Proliferative 用红色 (Hot/High Score)，Mesenchymal 用蓝色
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
  filter(Subtype != "SKCM.-") # 去掉无信息的

# 简化亚型名字，让图好看点
skcm_data <- skcm_data %>%
  mutate(Subtype_Label = case_when(
    grepl("BRAF", Subtype) ~ "BRAF Mutant",
    grepl("RAS", Subtype) ~ "RAS Mutant",
    grepl("NF1", Subtype) ~ "NF1 Mutant",
    grepl("Triple_WT", Subtype) ~ "Triple WT",
    TRUE ~ Subtype
  ))

# 设定顺序：WT (Low) -> BRAF (High)
skcm_data$Subtype_Label <- factor(skcm_data$Subtype_Label, 
                                  levels = c("Triple WT", "NF1 Mutant", "RAS Mutant", "BRAF Mutant"))

ggplot(skcm_data, aes(x = Subtype_Label, y = Predicted_IC50, fill = Subtype_Label)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.2) +
  scale_fill_brewer(palette = "YlOrRd") + # 渐变色
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

ggplot(prad_data, aes(x = Group, y = Predicted_IC50, fill = Group)) +
  geom_boxplot(alpha = 0.7) +
  stat_compare_means(method = "wilcox.test") +
  labs(
    title = "Validation in Prostate Cancer (PRAD)",
    subtitle = "ERG Fusion status check",
    x = "Subtype Group",
    y = "Predicted Score (IC50)"
  ) +
  theme_bw()
  

# ==============================================================================
# 1. 定义及计算 CBI (保持不变)
# ==============================================================================

# =============================================================================
# Section 6: CBI (Cytotoxic Burden Index) Subtype Validation
# =============================================================================
cytotoxic_drugs <- c(
  "Vinblastine", "Cisplatin", "Cytarabine", "Docetaxel", "Methotrexate", 
  "5-Fluorouracil", "Paclitaxel", "Irinotecan", "Oxaliplatin", 
  "Temozolomide", "Epirubicin", "Cyclophosphamide", "Mitoxantrone", 
  "Dactinomycin", "Bleomycin", "Dacarbazine", "Bleomycin (50 uM)"
)

# 计算 CBI
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
# 2. 合并亚型数据
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
# 3. 绘图函数 (通用)
# ==============================================================================
plot_cbi_validation_auto <- function(data, cancer_code, subtype_order, title_suffix) {
  
  # 1. 筛选数据
  plot_data <- data %>%
    filter(CancerCode == cancer_code) %>%
    filter(Subtype %in% subtype_order)
  
  # 2. 因子化排序
  plot_data$Subtype <- factor(plot_data$Subtype, levels = subtype_order)
  
  # 3. 自动计算显著的配对 (p < 0.05)
  # 使用 ggpubr 的 compare_means 函数计算所有两两比较
  stat_res <- compare_means(CBI ~ Subtype, data = plot_data, method = "wilcox.test")
  
  # 筛选出 p < 0.05 的行
  sig_stats <- stat_res %>% filter(p < 0.05)
  
  # 构建 comparisons 列表
  if (nrow(sig_stats) > 0) {
    significant_comparisons <- lapply(1:nrow(sig_stats), function(i) {
      c(sig_stats$group1[i], sig_stats$group2[i])
    })
  } else {
    significant_comparisons <- NULL # 如果没有显著差异，就不画线
  }
  
  # 4. 绘图
  p <- ggplot(plot_data, aes(x = Subtype, y = CBI, fill = Subtype)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.8) +
    geom_jitter(width = 0.2, alpha = 0.1, size = 0.8) +
    
    # 全局 P 值 (Kruskal-Wallis)
    # 调整位置到最高点的上方
    stat_compare_means(label.y = max(plot_data$CBI) * 1.5, size = 4) +
    
    # 配色
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
  
  # 5. 只有存在显著差异时才添加连线
  if (!is.null(significant_comparisons)) {
    p <- p + stat_compare_means(
      comparisons = significant_comparisons,
      method = "wilcox.test",
      label = "p.signif",
      size = 4,
      step.increase = 0.1,  # 自动调整每条线的高度间隔，防止重叠
      tip.length = 0.01
    )
  }
  
  return(p)
}

# ==============================================================================
# 4. 执行验证 (Run Validation)
# ==============================================================================

# --- A. BRCA: Basal (High Proliferation) ---
# 预期：Basal 最高
p1 <- plot_cbi_validation_auto(
  merged_cbi, 
  "BRCA", 
  c("BRCA.LumA", "BRCA.LumB", "BRCA.Her2", "BRCA.Basal"),
  ""
)
print(p1)

# --- B. OV: Proliferative vs Mesenchymal ---
# 注意：CancerCode 通常是 "OV" 而不是 "OVCA" (虽然亚型名字带OVCA)
p2 <- plot_cbi_validation_auto(
  merged_cbi,
  "OVCA", 
  c("OVCA.Mesenchymal", "OVCA.Differentiated", "OVCA.Immunoreactive", "OVCA.Proliferative"),
  ''
)
print(p2)

# --- C. PRAD: 亚型验证 (New) ---
# PRAD 亚型主要由融合基因驱动。
# PRAD.1-ERG 是最常见的融合，通常与更活跃的雄激素信号和代谢有关。
# 我们比较 ERG Fusion vs Others。
# 预期：ERG 融合型可能表现出较高的 CBI（如果模型捕捉到了其增殖特征）。

# 1. 准备 PRAD 亚型顺序
prad_subtypes <- c("PRAD.8-other", "PRAD.5-SPOP", "PRAD.2-ETV1", "PRAD.1-ERG")

p3 <- plot_cbi_validation_auto(
  merged_cbi,
  "PRAD",
  prad_subtypes,
  ""
)
print(p3)

# --- D. SKCM: 基因型验证 (New) ---
# SKCM 亚型基于体细胞突变。
# BRAF/RAS 突变通常驱动 MAPK 通路持续激活，导致高增殖。
# Triple WT (野生型) 通常增殖较慢或驱动力较弱。
# 预期：BRAF/RAS Mutants 的 CBI 显著高于 Triple WT。

# 1. 准备 SKCM 亚型顺序 (从低恶性到高恶性)
skcm_subtypes <- c("SKCM.Triple_WT", "SKCM.NF1_Any_Mutants", "SKCM.RAS_Hotspot_Mutants", "SKCM.BRAF_Hotspot_Mutants")

p4 <- plot_cbi_validation_auto(
  merged_cbi,
  "SKCM",
  skcm_subtypes,
  "BRAF/RAS Mutants (High Proliferation) show higher CBI than WT"
)
print(p4)

# --- E. 组合展示 ---
(p1 | p2) / (p3 | p4)



# ==============================================================================
# 1. 数据准备
# ==============================================================================

# 假设你已经有了 clean_subtypes (包含 BRCA 亚型) 和 est_clean (包含 ESTIMATE 分数)
# 如果没有 est_clean，请先运行之前的转置代码

# 1.1 准备 BRCA 亚型数据

# =============================================================================
# Section 7: Immune Score & Tumor Purity by BRCA Subtype
# =============================================================================
brca_subtypes <- clean_subtypes %>%
  filter(CancerCode == "BRCA") %>%
  filter(Subtype %in% c("BRCA.LumA", "BRCA.LumB", "BRCA.Her2", "BRCA.Basal")) %>%
  # 统一 ID 格式以便合并 (前15位)
  mutate(SampleID_Short = substr(SampleID, 1, 15),
         Subtype = ifelse(Subtype=='BRCA.Basal','Basal','Other'))

# 1.2 准备 ESTIMATE 数据
est_data_ready <- est_clean %>%
  mutate(SampleID_Short = substr(SampleID, 1, 15)) %>%
  mutate(SampleID_Short = gsub("\\.", "-", SampleID_Short))

# 1.3 合并
brca_est_merged <- brca_subtypes %>%
  inner_join(est_data_ready, by = "SampleID_Short")

# 1.4 设定绘图顺序 (从低恶性到高恶性)
brca_est_merged$Subtype <- factor(brca_est_merged$Subtype, 
                                  levels = c("Basal","Other"))

# ==============================================================================
# 2. 绘图验证：免疫评分 (Immune Score)
# ==============================================================================
# 预期：Basal 应该显著高于 LumA/LumB

p_immune <- ggplot(brca_est_merged, aes(x = Subtype, y = ImmuneScore, fill = Subtype)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.8) +
  geom_jitter(width = 0.2, alpha = 0.1, size = 0.5) +
  
  # 统计检验：重点比较 LumA vs Basal
  stat_compare_means(comparisons = list(c("Other", "Basal")), 
                     label = "p.signif", size=5) +
  stat_compare_means(label.y = max(brca_est_merged$ImmuneScore) * 1.1) +
  
  scale_fill_brewer(palette = "RdBu", direction = -1) + # 红色给 Basal
  labs(
    title = "High Immune Infiltration in Basal Subtype",
    subtitle = "Higher ImmuneScore = More immune cells (Potential sensitivity signal)",
    y = "ESTIMATE Immune Score",
    x = ""
  ) +
  theme_classic() +
  theme(legend.position = "none", axis.text.x = element_blank()) # 下面拼图时再显示轴标签

# ==============================================================================
# 3. 绘图验证：肿瘤纯度 (Tumor Purity)
# ==============================================================================
# 预期：Basal 的纯度可能被免疫细胞稀释，导致不如预期那么高，甚至低于 LumB/Her2

p_purity <- ggplot(brca_est_merged, aes(x = Subtype, y = TumorPurity, fill = Subtype)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.8) +
  geom_jitter(width = 0.2, alpha = 0.1, size = 0.5) +
  stat_compare_means(comparisons = list(c("Other", "Basal")),
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
# 4. 组合图表
# ==============================================================================

# 上下拼图
combined_plot <- p_immune / p_purity
print(combined_plot)


# =============================================================================
# Section 8: CCLE Primary vs Metastatic IC50
# =============================================================================
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
ggplot(plot_ccle_ic50_dist, aes(x = Status, y = IC50, fill = Status)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.8) +
  geom_jitter(width = 0.2, alpha = 0.1, size = 0.5) +
  stat_compare_means(comparisons = list(c('Metastatic', 'Primary')),label = "p.signif", size=5) +
  
  scale_fill_brewer(palette = "RdBu", direction = -1) +
  labs(
    title = "Cell line sensitivity (Vinblastine)",
    y = "lnIC50",
    x = "CCLE status"
  ) +
  theme_classic() +
  theme(legend.position = "none", axis.text.x = element_text(face="bold", size=11))

####################### classification

# 安装必要的包 (如果尚未安装)
# install.packages(c("mlr3", "mlr3learners", "mlr3viz", "ranger", "ggplot2", "tidyr", "dplyr"))


# ==============================================================================
# 1. 数据准备 (Data Preparation)
# ==============================================================================

# 1.1 准备标签 (Target: Subtype)

# =============================================================================
# Section 9: Machine Learning Classification (Subtype Prediction)
# =============================================================================
target_labels <- clean_subtypes %>%
  filter(CancerCode == "BRCA") %>%
  filter(Subtype %in% c("BRCA.LumA", "BRCA.LumB", "BRCA.Her2", "BRCA.Basal")) %>%
  dplyr::select(SampleID, Subtype) %>%
  mutate(SampleID = substr(SampleID, 1, 15)) %>%
  mutate(SampleID = gsub("\\.", "-", SampleID)) %>%
  mutate(Subtype = as.factor(Subtype)) %>% # mlr3 要求分类目标必须是 factor
  distinct(SampleID, .keep_all = TRUE)

# 1.2 准备 Feature Set A: 药物预测值 (Your Model)
# 使用之前定义的 cytotoxic_drugs 列表
drug_features <- pred_full %>%
  filter(DrugName %in% cytotoxic_drugs) %>%
  dplyr::select(SampleID, DrugName, Predicted_IC50) %>%
  mutate(SampleID = substr(SampleID, 1, 15)) %>%
  mutate(SampleID = gsub("\\.", "-", SampleID)) %>%
  pivot_wider(names_from = DrugName, values_from = Predicted_IC50)

# 1.3 准备 Feature Set B: Tumor Purity (Baseline)
purity_feature <- est_clean %>%
  dplyr::select(SampleID, TumorPurity) %>%
  mutate(SampleID = substr(SampleID, 1, 15)) %>%
  mutate(SampleID = gsub("\\.", "-", SampleID))

# 1.4 构建两个独立的数据集用于建模
# Dataset 1: 仅包含 Purity
data_purity <- target_labels %>%
  inner_join(purity_feature, by = "SampleID") %>%
  dplyr::select(-SampleID) %>% # 移除 ID 列，只留特征和标签
  na.omit()

# Dataset 2: 包含药物预测值
data_comb <- target_labels %>%
  inner_join(drug_features, by = "SampleID") %>%
  dplyr::select(-SampleID) %>%
  # 使用 unnest() 来展开所有 list 类型的列
  unnest(cols = everything(), keep_empty = TRUE) %>% # 展开所有列
  na.omit()

print(paste("Samples in Purity Task:", nrow(data_purity)))
print(paste("Samples in Comb Task:", nrow(data_comb)))


# ==============================================================================
# 2. 定义任务 (Tasks)
# ==============================================================================

# Task 1: Baseline (Purity)
task_purity <- as_task_classif(data_purity, target = "Subtype", id = "Baseline (Purity)")

# Task 2: Your Model (Comb Drugs)
task_comb <- as_task_classif(data_comb, target = "Subtype", id = "Comb Model (Drug Scores)")

# 设定分层采样 (Stratification)
# 这一步很重要，确保每一折里各亚型的比例与总体一致
task_purity$col_roles$stratum <- "Subtype"
task_comb$col_roles$stratum <- "Subtype"

# ==============================================================================
# 3. 定义学习器 (Learner) 与 重抽样策略 (Resampling)
# ==============================================================================

# 使用 Random Forest (ranger)
# importance = "impurity" 用于后续画特征重要性
learner <- lrn("classif.ranger", predict_type = "prob", importance = "impurity")

# 5折交叉验证
resampling <- rsmp("cv", folds = 5)

# ==============================================================================
# 4. 执行 Benchmark (核心对比)
# ==============================================================================

# 构建设计矩阵：两个任务，用同一个 learner，同样的 resampling
design <- benchmark_grid(
  tasks = list(task_purity, task_comb),
  learners = learner,
  resamplings = resampling
)

# 开始运行 (设置随机数种子保证复现)
set.seed(42)
bmr <- benchmark(design)

# 查看汇总结果
print(bmr$aggregate(msr("classif.acc")))

autoplot(bmr, measure = msr("classif.acc")) +
  labs(
    title = "Stratification accuracy",
    y = "ACC (5-fold CV)",
    x = ""
  ) +
  theme_classic() +
  theme(axis.text.x = element_text(face = "bold", size = 0))

# ==============================================================================
# 6. 绘制 Comb 模型的混淆矩阵热图
# ==============================================================================

# 提取 Comb 任务的预测结果
res_comb <- bmr$resample_result(2) # 索引2对应 task_comb (根据 grid 顺序)
cm <- res_comb$prediction()$confusion

# 转换为数据框以便 ggplot 绘图
cm_df <- as.data.frame(cm)
names(cm_df) <- c("True_Class", "Predicted_Class", "Freq")

# 绘图
ggplot(cm_df, aes(x = True_Class, y = Predicted_Class, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 5) +
  scale_fill_gradient(low = "#F0F0F0", high = "#DC143C") +
  labs(
    title = "Confusion Matrix: Comb Model",
    x = "Actual Clinical Subtype",
    y = "Predicted Subtype"
  ) +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 10),
    plot.title = element_text(face = "bold")
  )

# ==============================================================================
# 7. 特征重要性分析
# ==============================================================================

# 重新在全部数据上训练一个模型以提取 Importance
final_model <- lrn("classif.ranger", importance = "impurity")
final_model$train(task_comb)

# 提取重要性并绘图
importance_data <- as.data.frame(final_model$importance())
names(importance_data) <- "Importance"
importance_data$Feature <- rownames(importance_data)

# 取 Top 15
top_features <- importance_data %>%
  arrange(desc(Importance)) %>%
  head(15)

ggplot(top_features, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Top predictive features",
    x = "Drug Name",
    y = "Importance (Gini Impurity)"
  ) +
  theme_classic()


####survival

# ==============================================================================
# 1. 修改数据获取函数 (增加 Age, Gender, Stage)
# ==============================================================================

# =============================================================================
# Section 10: Survival Analysis (KM + Cox Regression)
# =============================================================================
get_survival_data_extended <- function(cancer_codes) {
  all_clin <- list()
  for (proj in cancer_codes) {
    proj_id <- paste0("TCGA-", proj)
    # 获取临床数据
    clin <- GDCquery_clinic(project = proj_id, type = "clinical", save.csv = FALSE)
    
    # 检查列名是否存在，防止报错 (不同癌种列名可能微调)
    cols_to_select <- c("submitter_id", "vital_status", "days_to_death", 
                        "days_to_last_follow_up", "gender", "age_at_index", 
                        "ajcc_pathologic_stage")
    
    # 提取存在的列
    valid_cols <- intersect(cols_to_select, colnames(clin))
    
    clin_sub <- clin %>%
      select(all_of(valid_cols)) %>%
      mutate(CancerCode = proj)
    
    all_clin[[proj]] <- clin_sub
  }
  bind_rows(all_clin)
}

# 获取数据
my_cancers <- c("LGG", "BRCA", "PRAD", "LUAD", "SKCM") 
survival_raw <- get_survival_data_extended(my_cancers)

# ==============================================================================
# 2. 数据清洗 (处理生存时间、年龄、分期)
# ==============================================================================
survival_clean <- survival_raw %>%
  dplyr::rename(PatientID = submitter_id) %>%
  mutate(
    # --- A. 生存时间 ---
    OS_Status = ifelse(vital_status == "Dead", 1, 0),
    OS_Time = ifelse(OS_Status == 1, days_to_death, days_to_last_follow_up),
    OS_Time_Months = OS_Time / 30.4,
    
    # --- B. 年龄 (确保数值型) ---
    # 有些数据里是 'age_at_index'，有些是 'age_at_diagnosis'
    Age = as.numeric(age_at_index), 
    
    # --- C. 分期简化 (Stage I-IV) ---
    # TCGA 分期很乱 (如 Stage IIA, Stage IIB)，统一简化
    Stage_Simple = case_when(
      grepl("Stage IV", ajcc_pathologic_stage, ignore.case = T) ~ "Stage IV",
      grepl("Stage III", ajcc_pathologic_stage, ignore.case = T) ~ "Stage III",
      grepl("Stage II", ajcc_pathologic_stage, ignore.case = T) ~ "Stage II",
      grepl("Stage I", ajcc_pathologic_stage, ignore.case = T) ~ "Stage I",
      TRUE ~ NA_character_ # 未知或无分期
    )
  ) %>%
  filter(!is.na(OS_Time) & OS_Time > 0)

# ==============================================================================
# 3. 合并数据 (CBI + Clinical + Subtype)
# ==============================================================================
# 假设 cbi_df 和 clean_subtypes 已经存在 (沿用之前的代码)

surv_analysis_data <- cbi_df %>%
  mutate(PatientID = substr(SampleID, 1, 12)) %>%
  inner_join(survival_clean, by = "PatientID") %>%
  # 还是建议加上亚型，做亚组分析有用
  left_join(clean_subtypes %>% mutate(PatientID = substr(SampleID, 1, 12)), by = "PatientID") %>%
  distinct(PatientID, .keep_all = TRUE) # 去重

print("Data merging complete. Columns available:")
print(colnames(surv_analysis_data))


# ==============================================================================
# 4. KM 绘图函数 (Optimal Cutoff 专用)
# ==============================================================================
plot_km_optimal <- function(data, cancer_code) {
  
  # 1. 筛选数据
  df_sub <- data %>% filter(CancerCode.x == cancer_code)
  
  if(nrow(df_sub) < 20) return(NULL)
  
  # 2. 寻找最佳截断点 (MaxStat) 
  res.cut <- surv_cutpoint(df_sub, time = "OS_Time_Months", event = "OS_Status", 
                           variables = "CBI", minprop = 0.2)
  
  cutoff_val <- res.cut$cutpoint$cutpoint
  
  # 3. 根据最佳点分类 (High vs Low)
  df_sub <- df_sub %>%
    mutate(Group = ifelse(CBI > cutoff_val, "High CBI", "Low CBI"))
  
  # 4. 拟合曲线
  fit <- survfit(Surv(OS_Time_Months, OS_Status) ~ Group, data = df_sub)
  
  # 5. 绘图
  ggsurvplot(
    fit, 
    data = df_sub,
    pval = TRUE,             
    pval.method = TRUE,      # 显示 Log-rank
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

# --- 执行画图 ---
# 1. LGG (预期非常显著)
p_lgg <- plot_km_optimal(surv_analysis_data, "LGG")
print(p_lgg)

# 2. BRCA (看看Optimal能否救回来)
p_brca <- plot_km_optimal(surv_analysis_data, "BRCA")
print(p_brca)

p_skcm <- plot_km_optimal(surv_analysis_data, "SKCM")
print(p_skcm)


# ==============================================================================
# 5. 多因素 Cox 回归与森林图 (Forest Plot)
# ==============================================================================

run_multivariate_cox <- function(data, cancer_code) {
  
  # 1. 筛选并清洗特定癌种的数据
  df_cox <- data %>% 
    filter(CancerCode.x == cancer_code) %>%
    filter(!is.na(Age)) # 去除年龄缺失
  
  # 2. 构建公式
  # 注意：LGG 通常没有 Stage，只有 Grade (但在 CDR 里可能也没有标准 Grade 列)
  # 所以我们根据癌种动态调整公式
  
  if (cancer_code == "LGG") {
    # LGG 只有 Age 和 Gender 可用 (或者你可以去合并 Grade 数据)
    formula_cox <- as.formula("Surv(OS_Time_Months, OS_Status) ~ CBI + Age + gender")
  } else {
    # BRCA 等通常有 Stage
    # 过滤掉 Stage 缺失的样本
    df_cox <- df_cox %>% filter(!is.na(Stage_Simple))
    formula_cox <- as.formula("Surv(OS_Time_Months, OS_Status) ~ CBI + Age + gender + Stage_Simple")
  }
  
  print(paste("Running Cox for:", cancer_code, "with N =", nrow(df_cox)))
  
  # 3. 运行 Cox 模型
  res.cox <- coxph(formula_cox, data = df_cox)
  
  # 打印简报
  print(summary(res.cox))
  
  # 4. 绘制森林图 (Forest Model)
  # 这是一个非常漂亮的出版级图表
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

# --- 执行 Cox 分析 ---

# 1. LGG 森林图
# 预期：CBI 的 Hazard Ratio (HR) 显著 > 1 (预后差)
# 或者是 < 1 (如果 CBI 反映的是化疗敏感获益)
forest_lgg <- run_multivariate_cox(surv_analysis_data, "LGG")
print(forest_lgg)

# 2. BRCA 森林图
# 看看校正了 Stage 之后，CBI 是否变得显著
forest_brca <- run_multivariate_cox(surv_analysis_data, "BRCA")
print(forest_brca)




# ==============================================================================
# 1. 定义药物分组 (Drug Sets)
# ==============================================================================
# 将药物按机制分组，同时也保留 "All"

# =============================================================================
# Section 11: CBI Variants (Mean, PCA, FA) & Survival
# =============================================================================
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
# 2. 定义 CBI 计算核心函数 (支持 Mean, PCA, FA)
# ==============================================================================
calculate_cbi_variant <- function(pred_data, drug_list, method = "Mean") {
  
  # 1. 筛选药物
  subset_data <- pred_data %>%
    filter(DrugName %in% drug_list) %>%
    select(SampleID, DrugName, Predicted_IC50)
  
  # 2. 转为宽格式 (Sample x Drug)
  wide_data <- subset_data %>%
    mutate(SampleID = substr(SampleID, 1, 15)) %>% # 统一 ID
    # 处理重复值：取平均
    group_by(SampleID, DrugName) %>%
    summarise(Predicted_IC50 = mean(Predicted_IC50, na.rm=TRUE), .groups="drop") %>%
    pivot_wider(names_from = DrugName, values_from = Predicted_IC50) %>%
    tibble::column_to_rownames("SampleID") %>%
    na.omit() # PCA/FA 不能有缺失值
  
  # 如果样本太少或药物太少，无法计算
  if (nrow(wide_data) < 10 || ncol(wide_data) < 2) return(NULL)
  
  # 3. 根据方法计算 Score
  scores <- tryCatch({
    if (method == "Mean") {
      # Z-score 后取平均
      scaled_data <- scale(wide_data)
      rowMeans(scaled_data, na.rm = TRUE)
      
    } else if (method == "PCA") {
      # 取第一主成分 (PC1)
      # PC1 通常捕获最大的变异方向（即主要的"耐药/敏感"轴）
      pca_res <- prcomp(wide_data, center = TRUE, scale. = TRUE)
      pca_res$x[, 1] 
      
    } else if (method == "FA") {
      # 因子分析 (Factor Analysis)，取第一因子
      # 相比 PCA，FA 更侧重于提取潜在的公共因子
      fa_res <- factanal(wide_data, factors = 1, scores = "regression")
      fa_res$scores[, 1]
    }
  }, error = function(e) return(NULL))
  
  if (is.null(scores)) return(NULL)
  
  # 4. 整理返回
  result <- data.frame(SampleID = names(scores), CBI = scores) %>%
    mutate(SampleID = gsub("\\.", "-", SampleID)) # ID 格式修正
  
  return(result)
}

# ==============================================================================
# 3. 循环遍历与评估 (Loop & Evaluation)
# ==============================================================================

# 设定要测试的癌种
target_cancers <- c("LGG", "BRCA", "PRAD", "LUAD", "SKCM") 

# 存储结果的容器
results_table <- data.frame()

# 循环：癌种 -> 药物组合 -> 计算方法
for (cancer in target_cancers) {
  
  # 获取该癌种的生存数据 (使用之前定义的 survival_clean)
  surv_cancer <- survival_clean %>% 
    filter(CancerCode == cancer) %>%
    mutate(SampleID_Join = substr(PatientID, 1, 12)) # 准备好用于连接的ID
  
  if(nrow(surv_cancer) < 20) next
  
  for (set_name in names(drug_sets_list)) {
    drugs <- drug_sets_list[[set_name]]
    
    for (method in c("Mean", "PCA", "FA")) {
      
      # A. 计算 CBI
      # 注意：pred_full 需要只包含该癌种的数据，或者全量计算后合并
      # 为了速度，建议先筛选出该癌种的预测值 (如果 pred_full 里有 CancerCode)
      # 这里假设 pred_full 包含所有
      
      cbi_res <- calculate_cbi_variant(pred_full, drugs, method)
      
      if (is.null(cbi_res)) next
      
      # B. 合并生存数据
      analysis_set <- cbi_res %>%
        mutate(SampleID_Join = substr(SampleID, 1, 12)) %>%
        inner_join(surv_cancer, by = "SampleID_Join")
      
      if (nrow(analysis_set) < 20) next
      
      # C. 运行单因素 Cox 回归 (Continuous Variable)
      # 我们看 CBI 作为一个连续值是否显著
      cox_fit <- tryCatch({
        coxph(Surv(OS_Time_Months, OS_Status) ~ CBI, data = analysis_set)
      }, error = function(e) NULL)
      
      if (!is.null(cox_fit)) {
        tidy_res <- tidy(cox_fit)
        
        # D. 记录结果
        results_table <- rbind(results_table, data.frame(
          Cancer = cancer,
          Drug_Set = set_name,
          Method = method,
          HR = exp(tidy_res$estimate), # 风险比
          P_Value = tidy_res$p.value,
          C_Index = summary(cox_fit)$concordance[1], # 一致性指数
          N_Samples = nrow(analysis_set)
        ))
      }
    }
  }
}

# ==============================================================================
# 4. 展示最佳策略 (Best Performers)
# ==============================================================================

# 按 P 值排序，查看每个癌种的最佳策略
best_strategies <- results_table %>%
  group_by(Cancer) %>%
  dplyr::filter(HR>1) 
#  arrange(P_Value) %>%
#  slice_head(n = 3) # 每个癌种看前3名

print(best_strategies)


run_cbi_survival_analysis <- function(cancer_code, 
                                      drug_set_name, 
                                      method = "FA", 
                                      flip_sign = FALSE) {
  
  message(paste0(">>> Analyzing: ", cancer_code, " | Drug Set: ", drug_set_name, " | Method: ", method))
  
  # 1. 获取药物列表
  if (!drug_set_name %in% names(drug_sets_list)) {
    stop("Drug set name not found in 'drug_sets_list'.")
  }
  drugs <- drug_sets_list[[drug_set_name]]
  
  # 2. 计算 CBI
  # 注意：这里假设 calculate_cbi_variant 和 pred_full 已经在环境中定义
  cbi_df <- calculate_cbi_variant(pred_full, drugs, method)
  
  if (is.null(cbi_df)) {
    warning("CBI calculation failed (not enough data/drugs).")
    return(NULL)
  }
  
  # 3. 符号翻转 (针对 PCA/FA 方向不确定的情况)
  if (flip_sign) {
    message("Note: Flipping CBI sign (-CBI).")
    cbi_df$CBI <- -cbi_df$CBI
  }
  
  # 4. 数据合并与清洗
  # 假设 survival_clean 已经在环境中
  plot_data <- cbi_df %>%
    mutate(SampleID_Join = substr(SampleID, 1, 12)) %>%
    inner_join(survival_clean, by = c("SampleID_Join" = "PatientID")) %>%
    mutate(CancerCode.x = CancerCode) # 为了适配之前定义的绘图函数
  
  # 检查样本量
  n_samples <- nrow(plot_data %>% filter(CancerCode.x == cancer_code))
  if (n_samples < 20) {
    warning(paste("Not enough samples for", cancer_code))
    return(NULL)
  }
  
  # 5. 生成图表
  # A. KM Curve (Optimal Cutoff)
  # 假设 plot_km_optimal 已经在环境中
  p_km <- plot_km_optimal(plot_data, cancer_code)
  
  # B. Cox Forest Plot
  # 假设 run_multivariate_cox 已经在环境中
  p_forest <- run_multivariate_cox(plot_data, cancer_code)
  
  # 6. 返回结果列表
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
# 1. 准备数据
# ==============================================================================

# A. 计算 CBI
# 微管抑制剂（紫杉醇）是乳腺癌化疗基石

# =============================================================================
# Section 12: BRCA Subtype-Specific Survival
# =============================================================================
cbi_brca <- calculate_cbi_variant(pred_full, drug_sets_list[["Antimetabolites"]], "FA")

# B. 检查方向 (重要！)
# 假设 High CBI = High Proliferation = Aggressive
# 如果 PCA 方向反了，记得 flip_sign = TRUE (这里先假设不翻转，你需要根据 Forest Plot 调整)
# cbi_brca$CBI <- -cbi_brca$CBI 

# C. 合并生存和亚型数据
plot_data_brca <- cbi_brca %>%
  mutate(SampleID_Join = substr(SampleID, 1, 12)) %>%
  inner_join(survival_clean, by = c("SampleID_Join" = "PatientID")) %>%
  filter(CancerCode== "BRCA") %>%
  inner_join(clean_subtypes %>% mutate(SampleID_12 = substr(SampleID,1,12)), by = c("SampleID_Join" = "SampleID_12"))# 包含 Subtype


# ==============================================================================
# 2. 定义分亚型绘图函数
# ==============================================================================
run_subtype_analysis <- function(data, target_subtype) {
  
  # 筛选特定亚型
  df_sub <- data %>% 
    filter(Subtype == target_subtype)
  
  # 检查样本量
  if(nrow(df_sub) < 30) {
    message(paste("Not enough samples for", target_subtype))
    return(NULL)
  }
  
  # --- KM 曲线 (Optimal Cutoff) ---
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
    pval.method = TRUE,      # 显示 Log-rank
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
  # --- Cox 回归 ---
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
# 3. 执行分析
# ==============================================================================

# --- A. Basal-like (三阴性，化疗主要人群) ---
# 预期：最可能显著。High CBI 可能代表超高增殖/耐药难治，或者对化疗敏感获益。
res_basal <- run_subtype_analysis(plot_data_brca, "BRCA.Basal")
print(res_basal$km)
print(res_basal$forest)
print(summary(res_basal$cox))

# --- B. Luminal B (高恶性激素受体阳性，常需化疗) ---
# 预期：High CBI 预后差。
res_lumb <- run_subtype_analysis(plot_data_brca, "BRCA.LumB")
print(res_lumb$km)

#####


# =============================================================================
# Section 13: LGG Subtype Analysis & TOP2A Correlation
# =============================================================================
lgg_subtypes <- TCGAbiolinks::PanCancerAtlas_subtypes() %>%
  filter(cancer.type == "LGG") %>%
  select(
    SampleID = pan.samplesID,
    Subtype_DNAmeth # 這是核心亚型
  ) %>%
  mutate(SampleID = substr(SampleID, 1, 12)) %>%
  # 过滤掉 NA 亚型
  filter(!is.na(Subtype_DNAmeth) & Subtype_DNAmeth != "NA")

common_lgg_samples <- intersect(unique(substr(pred_full$SampleID, 1, 12)), lgg_subtypes$SampleID)
#transcriptome = read_delim("/Users/hechang/Documents/chenlab/CellHit_onlyCPU/data/transcriptomics/celligner_CCLE_TCGA.csv")
top2a_df = transcriptome[match(common_lgg_samples,(transcriptome$index %>% substr(.,1,12))), c('index', 'TOP2A')] %>%
  mutate(SampleID_12 = substr(index, 1, 12))

best_cbi_df = calculate_cbi_variant(pred_full,drug_sets_list[["Topoisomerase_Inhibitors"]], "FA") %>%
  mutate(CBI = -as.numeric(CBI))

lgg_master_data <- best_cbi_df %>%
  mutate(SampleID_12 = substr(SampleID, 1, 12)) %>%
  # 1. 合并亚型
  inner_join(lgg_subtypes, by = c("SampleID_12" = "SampleID")) %>%
  # 2. 合并生存 (用于热图注释)
  inner_join(survival_clean, by = c("SampleID_12" = "PatientID")) %>%
  left_join(top2a_df)


# 1. 为了图形美观，按 CBI 中位数对亚型进行排序
lgg_master_data$Subtype_DNAmeth <- reorder(
  lgg_master_data$Subtype_DNAmeth, 
  lgg_master_data$CBI, 
  FUN = median
)

# 2. 绘图
p_boxplot <- ggplot(lgg_master_data, aes(x = Subtype_DNAmeth, y = CBI, fill = Subtype_DNAmeth)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.8, width = 0.6) +
  geom_jitter(width = 0.2, alpha = 0.3, size = 1) +
  
  # 统计检验：比较最低组和最高组
  stat_compare_means(
    method = "wilcox.test", 
    label = "p.signif",
    ref.group = levels(lgg_master_data$Subtype_DNAmeth)[1], # 以CBI最低的组为基准
    label.y = max(lgg_master_data$CBI) * 1.05
  ) +
  # 全局 P 值
  stat_compare_means(label.y = max(lgg_master_data$CBI) * 1.2, size = 4) +
  
  # 颜色：使用 Magma，颜色越深代表分越高（越恶性）
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

print(p_boxplot)


# ==============================================================================
# Panel D: Mechanism Validation (Target Correlation)
# ==============================================================================

p_corr <- ggplot(lgg_master_data, aes(x = TOP2A, y = CBI)) +
  # 按亚型着色，可以看到不同亚型的分布团
  geom_point(aes(color = Subtype_DNAmeth), alpha = 0.7, size = 2) +
  
  # 全局回归线
  geom_smooth(method = "lm", color = "black", linetype = "dashed", fill = "grey80") +
  
  # 相关性系数
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

print(p_corr)


p_corr_faceted <- ggplot(lgg_master_data, aes(x = TOP2A, y = CBI)) +
  # 1. 散点：依然按亚型着色，好看
  geom_point(alpha = 0.8, size = 1.5) +
  
  # 2. 回归线：每个分面单独拟合
  geom_smooth(method = "lm", color = "black", linetype = "dashed", se = TRUE) +
  
  # 3. 相关性统计
  stat_cor(method = "pearson", label.x.npc = "left", label.y.npc = "top", size = 3.5) +
  
  # 4. 关键：分面显示
  facet_wrap(~Subtype_DNAmeth, scales = "free") + 
  
  scale_color_viridis_d(option = "magma", direction = -1) +
  
  labs(
    title = "Target Engagement: Subtype-Specific Analysis",
    x = "TOP2A Expression (log2 TPM)",
    y = "CBI Score"
  ) +
  theme_bw() + # 使用 bw 主题配合分面更好看
  theme(
    strip.background = element_rect(fill = "grey90"), # 分面标题背景
    strip.text = element_text(face = "bold"),         # 分面标题文字
    legend.position = "none"                          # 分面已有标题，图例可以去掉
  )

print(p_corr_faceted)


#crispr_data = read_delim("/Users/hechang/Documents/chenlab/data/23Q4/CRISPRGeneDependency.csv")
#colnames(crispr_data)[1] = ''

####drug comb

# =============================================================================
# Section 14: Drug Combination Co-sensitivity Analysis
# =============================================================================
drug_comb_nci = read_delim("/Users/hechang/Documents/chenlab/data/drug_comb_nci.csv")



# ==============================================================================
# 1. 构建映射字典 (Name Map)
# ==============================================================================
# 左边是 NCI 中的名字，右边是你 pred_full 中的名字
name_map <- c(
  # --- Direct Matches (直接匹配) ---
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
  
  # --- Proxies (药理学替代 - 关键步骤) ---
  "Doxorubicin"      = "Epirubicin",       # 蒽环类替代
  "Capecitabine"     = "5-Fluorouracil",   # 5-FU 前体
  "Carboplatin"      = "Cisplatin",        # 铂类替代
  "Vincristine"      = "Vinblastine",      # 长春花碱类替代
  "Ifosfamide"       = "Cyclophosphamide", # 烷化剂替代
  "Daunorubicin"     = "Epirubicin",       # 蒽环类替代
  
  # --- Missing / No Prediction (无法预测的药物) ---
  # 这些药物在你的模型中没有，且没有好的替代，设为 NA
  "Leucovorin"       = NA, # 辅助药
  "Bevacizumab"      = NA, # 抗体
  "Cetuximab"        = NA, # 抗体
  "Etoposide"        = NA, # 你的列表里没有 Etoposide，也没有好的同类替代(除了Top2抑制剂，但不完全一样)
  "Prednisone"       = NA, # 激素
  "Procarbazine"     = NA,
  "Mechlorethamine"  = NA,
  "Bortezomib"       = NA,
  "Dexamethasone"    = NA,
  "Melphalan"        = NA,
  "Busulfan"         = NA,
  'Gemcitabine'      = NA
)

# ==============================================================================
# 2. 清洗组合数据并计算 Regimen Score
# ==============================================================================

# 2.1 拆解 NCI 表格
comb_processed <- drug_comb_nci %>%
  # 拆分分号
  separate_rows(Drugs, sep = ";") %>%
  mutate(Drugs_Clean = str_trim(Drugs)) %>%
  # 映射名字
  mutate(Model_Drug = name_map[Drugs_Clean]) %>%
  # 过滤掉无法预测的药物
  filter(!is.na(Model_Drug)) %>%
  # 再次确认映射后的药物真的在你的预测结果里
  filter(Model_Drug %in% unique(pred_full$DrugName))

# 看看我们保留了多少药物
print("Mapped Drugs Coverage:")
print(table(comb_processed$Model_Drug))


# 定义映射函数
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
    
    # --- TARGET / Pediatric / Non-TCGA Cancers (基于你的列表) ---
    str_detect(disease_name, "neuroblastoma") ~ "NBL",
    str_detect(disease_name, "medulloblastoma") ~ "MB",
    str_detect(disease_name, "wilms tumor") ~ "WT",
    str_detect(disease_name, "osteosarcoma") ~ "OS",
    str_detect(disease_name, "rhabdomyosarcoma") ~ "RMS", # 包括 alveolar/embryonal
    str_detect(disease_name, "acute lymphoblastic leukemia") ~ "ALL",
    str_detect(disease_name, "ewing sarcoma") ~ "EWS",
    str_detect(disease_name, "hepatoblastoma") ~ "HB",
    str_detect(disease_name, "retinoblastoma") ~ "RB",
    str_detect(disease_name, "ependymoma") ~ "EPN",
    str_detect(disease_name, "meningioma") ~ "MNG",
    
    # --- Sarcomas (General) ---
    str_detect(disease_name, "sarcoma") ~ "SARC", # 捕获剩余的肉瘤
    
    # --- Fallback ---
    TRUE ~ "Other" # 无法识别的归为 Other
  )
}

# 1. 完善 tcga_md，添加 CancerCode
tcga_md_clean <- tcga_md %>%
  mutate(CancerCode = assign_cancer_code(disease)) %>%
  select(th_dataset_id, disease, CancerCode) # 只保留需要的列

# 检查一下映射结果，看看有没有大量的 "Other"
print("Distribution of Cancer Codes:")
print(table(tcga_md_clean$CancerCode))

# 2. 更新 pred_scaled 生成逻辑 (全量样本)
# 注意：你需要确认 pred_full 里的 SampleID 和 tcga_md 里的 th_dataset_id 是怎么对应的
# 假设 pred_full$SampleID 就是 th_dataset_id (或者需要简单处理)

pred_scaled_all <- pred_full %>%
  inner_join(tcga_md_clean, by = c("SampleID" = "th_dataset_id")) %>%
  group_by(DrugName) %>%
  mutate(Z_Score = scale(Predicted_IC50)) %>%
  ungroup() %>%
  select(SampleID, DrugName, Z_Score, CancerCode, disease)

print(paste("Original rows:", nrow(pred_full)))
print(paste("Merged rows:", nrow(pred_scaled_all)))

# 重新计算 Regimen Score
regimen_scores_all <- comb_processed %>% # 这是之前整理好的 NCI 组合-单药映射表
  inner_join(pred_scaled_all, by = c("Model_Drug" = "DrugName")) %>%
  
  group_by(SampleID, Name, Indication, CancerCode) %>%
  summarise(
    Regimen_Score = mean(Z_Score, na.rm = TRUE),
    Drug_Count = n_distinct(Model_Drug),
    .groups = "drop"
  ) %>%
  filter(Drug_Count >= 2)

# 查看现在的癌种覆盖情况 (应该比之前多很多)
print(table(regimen_scores_all$CancerCode))

# ==============================================================================
# Panel A: Co-sensitivity of Approved Pairs
# ==============================================================================
# 我们选择两组经典的组合：
# 1. BRCA: AC 方案 (Doxorubicin + Cyclophosphamide) -> 映射为 Epirubicin + Cyclophosphamide
# 2. COAD: FOLFOX 方案 (5-FU + Oxaliplatin)


# 1. 定义 NCI Indication -> TCGA CancerCode 的映射表
# 根据你的 drug_comb_nci 内容调整
indication_map <- list(
  "Breast Cancer" = "BRCA",
  "Colorectal Cancer" = c("COAD", "READ"), # 结直肠癌通常合并
  "Lung Cancer" = c("LUAD", "LUSC"),       # 肺癌合并腺癌和鳞癌
  "Ovarian Cancer" = "OV",
  "Testicular Cancer" = "TGCT",
  "Gastric Cancer" = "STAD",
  "Pancreatic Cancer" = "PAAD",
  "Hodgkin Lymphoma" = "DLBC", # TCGA 只有 DLBC，勉强对应
  "Non-Hodgkin Lymphoma" = "DLBC",
  "Urothelial Cancer" = "BLCA",
  "Neuroblastoma" = NA, # TCGA 无此癌种
  "Soft Tissue Sarcoma" = "SARC",
  "Myeloproliferative Neoplasms" = "LAML" # 对应白血病
)

# 2. 准备循环计算的数据
# 我们需要从 comb_processed 中获取方案及其包含的药物
regimens_to_test <- comb_processed %>%
  select(Name, Indication, Model_Drug) %>%
  distinct()

# 确保只保留有对应 TCGA 数据的方案
regimens_to_test <- regimens_to_test %>%
  rowwise() %>%
  mutate(Target_Cancers = list(indication_map[[Indication]])) %>%
  filter(!any(is.na(Target_Cancers))) %>%
  ungroup()

print(head(regimens_to_test))

results_df <- data.frame()

# 获取所有方案名称
unique_regimens <- unique(regimens_to_test$Name)

for (reg_name in unique_regimens) {
  
  # A. 获取该方案的信息
  sub_data <- regimens_to_test %>% filter(Name == reg_name)
  drugs <- unique(sub_data$Model_Drug)
  cancers <- unlist(unique(sub_data$Target_Cancers)) # 可能有多个 (如 LUAD, LUSC)
  
  # 至少要有2个药才能算相关性
  if (length(drugs) < 2) next
  
  # B. 生成所有可能的两两组合
  drug_pairs <- combn(drugs, 2, simplify = FALSE)
  
  # C. 在目标癌种中计算相关性
  for (cancer in cancers) {
    
    # 提取该癌种的预测数据
    cancer_pred <- pred_scaled_all %>%
      filter(CancerCode == cancer) %>%
      filter(DrugName %in% drugs) %>%
      select(SampleID, DrugName, Z_Score) %>%
      pivot_wider(names_from = DrugName, values_from = Z_Score)
    
    # 如果样本太少，跳过
    if (nrow(cancer_pred) < 20) next
    
    # 对每一对药计算 R 和 P
    for (pair in drug_pairs) {
      d1 <- pair[1]
      d2 <- pair[2]
      
      # 检查列是否存在
      if (!d1 %in% colnames(cancer_pred) || !d2 %in% colnames(cancer_pred)) next
      
      # 计算检验
      test <- cor.test(cancer_pred[[d1]], cancer_pred[[d2]], method = "pearson")
      
      # 记录结果
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

# D. 筛选最佳结果
# 规则：P < 0.05，按 R 从大到小排序
top_results <- results_df %>%
  filter(P < 0.05) %>%
  arrange(desc(R)) %>%
  # 可选：去重，防止同一个方案的多个配对霸榜，每个方案只取最好的一个 Pair
  group_by(Regimen) %>%
  slice_head(n = 1) %>%
  ungroup() %>%
  # 再按 R 排序
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
    
    # pivot_wider 后，如果某个药完全没有数据，列可能根本不会生成
    if (nrow(plot_dat) < 1) {
      message(paste("Skipping:", info$Regimen, "in", info$Cancer, "- Not enough matching samples."))
      next
    }
    
    if (!all(c(info$Drug1, info$Drug2) %in% colnames(plot_dat))) {
      message(paste("Skipping:", info$Regimen, "- One of the drugs is missing in the column names."))
      next
    }
    
    # 3. 绘图：使用 .data[[string]] 语法代替 aes_string
    # 这种写法可以完美处理 "5-Fluorouracil" 这种带横杠的名字，也不会报 object not found
    p <- ggplot(plot_dat, aes(x = .data[[info$Drug1]], y = .data[[info$Drug2]])) +
      geom_point(alpha = 0.4, color = "#2E8B57", size = 1.5) +
      geom_smooth(method = "lm", color = "black", linetype = "dashed", se = TRUE) +
      
      # 添加统计参数
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
    
    # 保存到列表
    plot_list_a[[plot_counter]] <- p
    plot_counter <- plot_counter + 1
  }
  
  # 4. 组合展示
  if (length(plot_list_a) > 0) {
    # 自动计算行列数
    final_panel_a <- ggarrange(plotlist = plot_list_a, 
                               ncol = min(length(plot_list_a), 2), 
                               nrow = ceiling(length(plot_list_a)/2))
    print(final_panel_a)
  } else {
    print("No valid plots were generated.")
  }
  
} else {
  print("No significant pairs found in 'best_4'.")
  final_panel_a <- NULL
}




# ===============================
# Panel B
#

# 1. 准备数据: FOLFOX in COAD
target_regimen <- c("5-Fluorouracil", "Oxaliplatin")
target_cancer <- "COAD"

waterfall_data <- pred_scaled_all %>%
  filter(CancerCode == target_cancer) %>%
  filter(DrugName %in% target_regimen) %>%
  select(SampleID, DrugName, Z_Score)

# 2. 计算每个病人的总分用于排序
rank_data <- waterfall_data %>%
  group_by(SampleID) %>%
  summarise(Total_Score = sum(Z_Score)) %>%
  arrange(desc(Total_Score))

# 3. 设置因子顺序 (让柱子按高低排列)
waterfall_data$SampleID <- factor(waterfall_data$SampleID, levels = rank_data$SampleID)

# 4. 绘图
panel_b <- ggplot(waterfall_data, aes(x = SampleID, y = Z_Score, fill = DrugName)) +
  geom_col(position = "stack", width = 1) + # 堆叠柱状图
  scale_fill_manual(values = c("5-Fluorouracil" = "#4682B4", "Oxaliplatin" = "#DC143C")) +
  
  # 添加一条线表示“敏感阈值” (例如 Z > 0)
  geom_hline(yintercept = 0, linetype="dashed", color="grey") +
  
  labs(
    title = "FOLFOX Response Stratification (COAD)",
    subtitle = "Identifying patients with dual sensitivity",
    x = "Patients",
    y = "Predicted Sensitivity (Z-Score)"
  ) +
  theme_classic() +
  theme(
    axis.text.x = element_blank(), # 隐藏X轴病人ID
    axis.ticks.x = element_blank(),
    legend.position = "top"
  )

print(panel_b)


# ==============================================================================
# Panel C: Regimen Specificity Heatmap
# ==============================================================================

# 1. 计算每个方案在每个癌种中的平均分
spec_data <- regimen_scores_all %>%
  group_by(Name, CancerCode) %>%
  summarise(Mean_Score = mean(Regimen_Score), .groups = "drop") %>%
  # 过滤样本量太少的癌种或方案以保持图表整洁
  group_by(CancerCode) %>%
  filter(n() > 3) %>% 
  ungroup()

# 2. 转矩阵
spec_mat <- spec_data %>%
  pivot_wider(names_from = CancerCode, values_from = Mean_Score) %>%
  tibble::column_to_rownames("Name") %>%
  as.matrix()

# 简单的 NA 填充 (填最小值，避免热图报错)
spec_mat[is.na(spec_mat)] <- min(spec_mat, na.rm = TRUE)

# 3. 绘制热图
# 我们希望看到对角线趋势
pheatmap(
  scale = "column",
  spec_mat,
  color = colorRampPalette(c("navy", "white", "firebrick"))(100),
  cluster_rows = TRUE,
  cluster_cols = TRUE,
  main = "Panel C: Regimen Specificity (Mean Predicted Sensitivity)",
  fontsize_row = 8,
  fontsize_col = 9,
  angle_col = 45
)


# 设定 Top N (例如 Top 600, 或 Top 10% 的病人)
TOP_N <- 600

# ==============================================================================
# 1. 提取每个药物的 Top N 敏感病人
# ==============================================================================
# 假设 High Z-Score = Sensitive (反向预测逻辑)
top_patients_list <- pred_scaled_all %>%
  group_by(DrugName) %>%
  # 对每个药，选 Z-score 最大的 600 人
  slice_max(order_by = Z_Score, n = TOP_N) %>%
  ungroup() %>%
  select(DrugName, SampleID)

# ==============================================================================
# 2. 计算药物两两之间的病人重叠数 (Intersection Matrix)
# ==============================================================================
# 使用矩阵乘法快速计算重叠 (A x A^T)
# 1. 转为 0/1 矩阵 (行=病人, 列=药物)
binary_mat <- table(top_patients_list$SampleID, top_patients_list$DrugName)
class(binary_mat) <- "matrix"
# 确保是二值 (虽然 slice_max 应该保证唯一，但防万一)
binary_mat[binary_mat > 0] <- 1 

# 2. 矩阵相乘得到重叠矩阵
# intersection_mat[i, j] = 药物i 和 药物j 共同的病人数
intersection_mat <- t(binary_mat) %*% binary_mat

# 3. 转为长格式数据框
overlap_df <- as.data.frame(as.table(intersection_mat)) %>%
  rename(Drug1 = Var1, Drug2 = Var2, Overlap_Count = Freq) %>%
  # 去除自相关
  filter(Drug1 != Drug2) %>%
  # 去除重复对 (只留一半)
  filter(as.character(Drug1) < as.character(Drug2)) %>%
  # 过滤掉重叠度太低的组合 (可选，比如少于 50 人的就不看了)
  filter(Overlap_Count > 100)

print(head(overlap_df))
# ==============================================================================
# 3. 构建注释字典 (Support Level)
# ==============================================================================

# 3.1 准备 Approved Pairs (已批准的组合)
# 复用之前的逻辑
approved_pairs_list <- comb_processed %>%
  select(Name, Model_Drug) %>%
  distinct() %>%
  inner_join(., ., by = "Name") %>%
  filter(Model_Drug.x < Model_Drug.y) %>%
  mutate(Pair_ID = paste(Model_Drug.x, Model_Drug.y, sep = "_")) %>%
  pull(Pair_ID) %>%
  unique()

# 3.2 准备 Drug Indications (药物适应症映射)
# 逻辑：如果一个药出现在 "Breast Cancer" 的某个方案里，它就拥有 "Breast Cancer" 标签
drug_indications_map <- comb_processed %>%
  select(Model_Drug, Indication) %>%
  distinct() %>%
  group_by(Model_Drug) %>%
  summarise(Indications = list(unique(Indication)))

# 3.3 为 overlap_df 打标签
# 这是一个逐行判断的过程
get_support_level <- function(d1, d2) {
  pair_id <- paste(d1, d2, sep = "_")
  
  # 1. 检查是否 Approved Combination
  if (pair_id %in% approved_pairs_list) {
    return("Approved Combination")
  }
  
  # 2. 检查是否 Shared Indication
  # 获取 d1 和 d2 的适应症列表
  ind1 <- drug_indications_map$Indications[drug_indications_map$Model_Drug == d1]
  ind2 <- drug_indications_map$Indications[drug_indications_map$Model_Drug == d2]
  
  if (length(ind1) > 0 && length(ind2) > 0) {
    # 取交集
    common <- intersect(unlist(ind1), unlist(ind2))
    if (length(common) > 0) {
      return("Sharing Indication")
    }
  }
  
  # 3. 既不是组合，也没有共同适应症
  return("Novel / Others")
}

# 应用函数 (使用 mapply 向量化操作)
overlap_df$Support_Level <- mapply(get_support_level, overlap_df$Drug1, overlap_df$Drug2)

# 设置因子顺序，让 Approved 绘图时在最上层
overlap_df$Support_Level <- factor(overlap_df$Support_Level, 
                                   levels = c("Novel / Others", "Sharing Indication", "Approved Combination"))

print(table(overlap_df$Support_Level))

novel_subset <- overlap_df %>% filter(Support_Level == "Novel / Others")
threshold <- quantile(novel_subset$Overlap_Count, 0.90) 

print(paste("Novel Threshold (Top 10% overlap count):", threshold))

# 更新 Support_Level 逻辑
overlap_df_refined <- overlap_df %>%
  mutate(Refined_Status = case_when(
    Support_Level == "Approved Combination" ~ "Approved Combination",
    Support_Level == "Sharing Indication" ~ "Sharing Indication",
    # 如果是 Novel 且重叠人数大于阈值 -> 定义为 Novel (High Potential)
    Support_Level == "Novel / Others" & Overlap_Count >= threshold ~ "Novel (High Overlap)",
    # 其他的 -> Others
    TRUE ~ "Others"
  ))

# 设定因子顺序 (决定图例顺序和绘图图层顺序)
# 我们希望 Others 在最底层，Approved 在最上层
order_levels <- c("Others", "Novel (High Overlap)", "Sharing Indication", "Approved Combination")
overlap_df_refined$Refined_Status <- factor(overlap_df_refined$Refined_Status, levels = order_levels)
# ==============================================================================
# 4. 绘图 (Refined Bubble Plot)
# ==============================================================================

# 定义新配色方案
custom_colors_refined <- c(
  "Approved Combination" = "#EE442F",  # 红色 (验证)
  "Sharing Indication"   = "#006400",  # 深绿 (同适应症)
  "Novel (High Overlap)" = "#4A4A4A",  # 深灰/黑 (发现 - 重点!)
  "Others"               = "#E0E0E0"   # 极浅灰 (背景噪音)
)

# 定义透明度 (让 Others 几乎隐形，突出重点)
custom_alpha_refined <- c(
  "Approved Combination" = 1,
  "Sharing Indication"   = 0.8,
  "Novel (High Overlap)" = 0.9,
  "Others"               = 0.3 
)

# 排序：确保画图时大球不会遮挡关键点，或者 Approved 在最上面
# 这里的逻辑是按照 Factor level 排序，Others 会先画
overlap_df_refined <- overlap_df_refined %>% arrange(Refined_Status)

p_refined <- ggplot(overlap_df_refined, aes(x = Drug1, y = Drug2)) +
  # 1. 网格背景
  geom_tile(fill = NA, color = "white") + # 稍微用白色隔开
  
  # 2. 气泡
  geom_point(aes(size = Overlap_Count, 
                 color = Refined_Status, 
                 alpha = Refined_Status)) +
  
  # 3. 颜色映射
  scale_color_manual(values = custom_colors_refined) +
  
  # 4. 透明度映射
  scale_alpha_manual(values = custom_alpha_refined) +
  
  # 5. 大小映射
  scale_size_continuous(range = c(1, 12), name = "Sample Overlap") +
  
  # 6. 主题调整
  theme_minimal() +
  theme(
    # X轴文字 (顶部)
    axis.text.x.top = element_text(angle = 45, hjust = 0, vjust = 0, size = 9, face = "bold"),
    axis.text.y = element_text(size = 9, face = "bold"),
    axis.title = element_blank(),
    
    # 图例优化
    legend.position = "right",
    legend.title = element_text(face = "bold"),
    panel.grid = element_blank(),
    
    # 增加一点边距防止文字被切
    plot.margin = margin(t = 10, r = 10, b = 10, l = 10)
  ) +
  scale_x_discrete(position = "top") +
  coord_fixed() +
  
  # 标题
  labs(title = "Drug Combination Discovery")

print(p_refined)


##### 词袋分析
target_samples = lgg_master_data$SampleID
res.cut <- surv_cutpoint(lgg_master_data, time = "OS_Time_Months", event = "OS_Status", 
                         variables = "CBI", minprop = 0.2)
cutoff_val <- res.cut$cutpoint$cutpoint

low_samples = lgg_master_data$SampleID[lgg_master_data$CBI<cutoff_val]
high_samples = lgg_master_data$SampleID[lgg_master_data$CBI>=cutoff_val]

sub_expr <- transcriptome[transcriptome$index %in% target_samples, -1] %>% 
  tibble::column_to_rownames('index')

# 1.2 将每个样本的表达量转为 Rank，并二值化 (Top 2000 = 1, Else = 0)
get_top2000_binary <- function(mat, n = 2000) {
  message(paste("Input matrix shape:", nrow(mat), "Samples x", ncol(mat), "Genes"))
  # 1. 对每一行 (样本) 进行 Ranking
  # apply(mat, 1, ...) 会对每一行操作
  # 关键点：R 的 apply 函数在对行操作后，会自动转置结果
  # 所以，ranks 的结果本身就会变成 [Genes x Samples]
  ranks <- apply(mat, 1, rank, ties.method = "first")
  # 2. 获取总基因数 (现在是 ranks 的行数)
  total_genes <- nrow(ranks)
  # 3. 二值化
  # 我们需要 Top N，也就是 Rank > (总数 - N)
  binary_mat <- ifelse(ranks > (total_genes - n), 1, 0)
  # 此时 binary_mat 已经是 [Genes x Samples]
  # 行名是基因，列名是样本ID
  return(binary_mat)
}
binary_top2000 <- get_top2000_binary(sub_expr, n = 2000)

mat_high <- binary_top2000[, high_samples]
mat_low <- binary_top2000[, low_samples]

count_high <- rowSums(mat_high) # 向量：每个基因在High组出现的次数
count_low <- rowSums(mat_low)   # 向量：每个基因在Low组出现的次数

n_high <- length(high_samples)
n_low <- length(low_samples)

# 构建结果数据框
diff_freq_df <- data.frame(
  Gene = rownames(binary_top2000),
  Count_High = count_high,
  Count_Low = count_low
) %>%
  # 过滤掉两边都没出现的基因 (加速计算)
  filter(Count_High > 0 | Count_Low > 0) %>%
  rowwise() %>%
  mutate(
    # Fisher Test
    P_Value = fisher.test(matrix(c(Count_High, n_high - Count_High, 
                                   Count_Low, n_low - Count_Low), nrow=2))$p.value,
    # Odds Ratio (为了防止除以0，加个极小值)
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

# 假设 gene_list 是你准备好的排序向量 (Log2OR 或其他指标)
genes_of_interest <- diff_freq_df %>%
  filter(Log2OR > 0.5 & P_Value < 0.05) %>%
  pull(Gene)

print(paste("Selected genes count:", length(genes_of_interest)))

# 2. 运行 ORA (enricher)
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
# 1. 建立“已知解释”的黑名单 (Blocklist)
# ==============================================================================

# 之前发现的显著通路
pathways_to_exclude <- c(
  "HALLMARK_E2F_TARGETS", "HALLMARK_G2M_CHECKPOINT", # 基础增殖
  "HALLMARK_KRAS_SIGNALING_UP", "HALLMARK_KRAS_SIGNALING_DN","HALLMARK IL6 JAK STAT3 SIGNALING", # 驱动信号
  "HALLMARK_INFLAMMATORY_RESPONSE", "HALLMARK_TNFA_SIGNALING_VIA_NFKB", # 免疫/炎症
  "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION", "HALLMARK_HYPOXIA" # 基质/耐药
)

# 提取这些通路里的所有基因
known_genes <- m_h %>% 
  filter(gs_name %in% pathways_to_exclude) %>% 
  pull(gene_symbol) %>% 
  unique()

print(paste("Number of 'Explained' Genes to exclude:", length(known_genes)))

# ==============================================================================
# 2. 筛选 Novel Markers
# ==============================================================================
# 条件：
# 1. 统计显著 (FDR < 0.05)
# 2. 效应量大 (|Log2OR| > 1)
# 3. 不在已知名单中 (!Gene %in% known_genes)

novel_df <- diff_freq_df %>%
  filter(FDR < 0.05, abs(Log2OR) > 1) %>% # 先选显著的
  filter(!Gene %in% known_genes) %>%      # 剔除已知通路基因
  mutate(Direction = ifelse(Log2OR > 0, "High CBI (Novel)", "Low CBI (Novel)"))

# 提取 Top Candidates (两边各取前 8 个)
top_novel_high <- novel_df %>% filter(Direction == "High CBI (Novel)") %>% arrange(desc(Log2OR)) %>% head(8)
top_novel_low  <- novel_df %>% filter(Direction == "Low CBI (Novel)") %>% arrange(Log2OR) %>% head(8)

top_novel_genes <- bind_rows(top_novel_high, top_novel_low)

print("Top Novel High-CBI Markers:")
print(top_novel_high$Gene)

# ==============================================================================
# 3. 准备绘图数据
# ==============================================================================
plot_data_novel <- diff_freq_df %>%
  mutate(
    # 定义分类
    Category = case_when(
      Gene %in% top_novel_high$Gene ~ "Novel gene (High CBI)",
      Gene %in% top_novel_low$Gene ~ "Novel gene (Low CBI)",
      Gene %in% known_genes ~ "Known Pathways", # 已知基因归为背景
      TRUE ~ "NS"
    ),
    # 定义透明度 (让 Novel 的点不透明，其他的透明)
    Alpha = ifelse(grepl("Novel", Category), 1, 0.3),
    # 定义大小
    Size = ifelse(grepl("Novel", Category), 2, 1)
  )

# ==============================================================================
# 4. 绘图
# ==============================================================================
# 自定义颜色
color_map <- c(
  "Novel gene (High CBI)" = "#DC143C", # 亮红
  "Novel gene (Low CBI)"  = "#4682B4", # 亮蓝
  "Known Pathways" = "black", # 灰色背景
  "NS" = "grey90"
)

max_val <- max(abs(diff_freq_df$Log2OR), na.rm = TRUE)
limit_x <- max_val * 1.1 

ggplot(plot_data_novel, aes(x = Log2OR, y = -log10(P_Value))) +
  # 1. 背景点 (NS 和 Known)
  geom_point(data = subset(plot_data_novel, !grepl("Novel", Category)),
             aes(color = Category, alpha = Alpha, size = Size)) +
  
  # 2. 重点点 (Novel) - 放在上层
  geom_point(data = subset(plot_data_novel, grepl("Novel", Category)),
             aes(color = Category, alpha = Alpha, size = Size), shape=19) +
  
  # 3. 标签 (只标 Novel)
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
# 0. 定义细胞注释和药物列表
# ==============================================================================

# Tissue 注释
annotations_tissue <- list(
  '0' = 'CD8_T', '1' = 'Naive_B', '2' = 'TAMs', '3' = 'T_reg',
  '4' = 'NK', '5' = 'Plasma_Cells', '6' = 'Pr_B', '7' = 'pDCs', '8' = 'Mast_Cells'
)

# Blood 注释
annotations_blood <- c(
  "0" = "NK", "1" = "Naive_T", "2" = "CD14+_Monocytes", "3" = "Naive_B",
  "4" = "gdT", "5" = "Pr_Lymphocytes", "6" = "Plasma_Cells", "7" = "Platelets",
  "8" = "pDCs", "9" = "HSPC"
)

# 细胞毒性药物列表 (微管抑制剂组)

target_drugs <- drug_sets_list$Microtubule_Inhibitors
#target_drugs = cytotoxic_drugs 

# ==============================================================================
# 1. 通用数据处理函数 (Process Data)
# ==============================================================================
process_scRNA_data <- function(pred_df, annotation_list, drug_filter) {
  
  # 转换注释列表为 DF
  if (is.list(annotation_list) && !is.atomic(annotation_list)) {
    anno_df <- tibble(Cluster = names(annotation_list), CellType = unlist(annotation_list))
  } else {
    anno_df <- tibble(Cluster = names(annotation_list), CellType = as.character(annotation_list))
  }
  
  # 数据清洗与合并
  data_clean <- pred_df %>%
    separate_wider_delim(SampleID, '-', names=c('Patient','Status','Cluster'), too_many = "merge") %>%
    mutate(Cluster = as.character(Cluster)) %>%
    left_join(anno_df, by = "Cluster") %>%
    filter(DrugName %in% drug_filter) %>%
    
    # 计算 CBI (标准化后取均值)
    group_by(DrugName) %>%
    mutate(Z_Score = scale(Predicted_IC50)) %>%
    ungroup() %>%
    group_by(Patient, Status, CellType, Cluster) %>%
    summarise(CBI = mean(Z_Score, na.rm = TRUE), .groups = "drop")
  
  return(data_clean)
}

# 处理两套数据
cbi_tissue <- process_scRNA_data(scRNA_pred_tissue, annotations_tissue, target_drugs)
cbi_blood  <- process_scRNA_data(scRNA_pred_blood, annotations_blood, target_drugs)


# ==============================================================================
# Plot Function A: Mechanism Validation (Cell Type Boxplot)
# ==============================================================================
plot_mechanism <- function(data, title_suffix) {
  
  # 排序：按中位数排序
  data$CellType <- reorder(data$CellType, data$CBI, FUN = median)
  
  p <- ggplot(data, aes(x = CellType, y = CBI, fill = CellType)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.8) +
    geom_jitter(width = 0.2, alpha = 0.4, size = 1.5) +
    stat_compare_means(label.y = max(data$CBI) * 1.1) +
    scale_fill_viridis_d(option = "turbo") +
    labs(
      title = paste0("Mechanism Validation: ", title_suffix),
      x = "Cell Type", y = "Score"
    ) +
    theme_classic() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
      legend.position = "none"
    )
  return(p)
}

# ==============================================================================
# Plot Function B: Longitudinal (Pre vs Post)
# ==============================================================================
plot_longitudinal <- function(data, target_cells, title_suffix) {
  
  # 筛选配对病人
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
  
  p <- ggplot(plot_data, aes(x = Group, y = CBI, fill = Group)) +
    geom_boxplot(alpha = 0.7, outlier.shape = NA) +
    geom_point(aes(group = Patient), size = 2, alpha = 0.6) +
    #geom_line(aes(group = Patient), color = "grey50", alpha = 0.5) +
    facet_wrap(~CellType, scales = "free_y") +
    #stat_compare_means(method = "t.test", paired = TRUE, label = "p.format", label.x.npc = "center") +
    scale_fill_manual(values = c("Pre-treatment" = "#DC143C", "Post-treatment" = "#4682B4")) +
    labs(title = paste0("Pre vs Post: ", title_suffix), y = "Score") +
    theme_bw() + theme(legend.position = "none", axis.title.x = element_blank())
  
  return(p)
}

# ==============================================================================
# Plot Function C: Baseline Prediction (Resistant vs Sensitive)
# ==============================================================================
plot_baseline <- function(data, target_cells, title_suffix) {
  
  # 定义结局
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
  
  p <- ggplot(plot_data, aes(x = Outcome, y = CBI, fill = Outcome)) +
    geom_boxplot(alpha = 0.7, outlier.shape = NA) +
    geom_jitter(width = 0.2, size = 2, alpha = 0.6) +
    facet_wrap(~CellType, scales = "free_y") +
    #stat_compare_means(method = "t.test", label = "p.signif", label.x.npc = "center") +
    scale_fill_manual(values = c("Sensitive" = "#2E8B57", "Resistant" = "#B22222")) +
    labs(title = paste0("Baseline Prediction: ", title_suffix), y = "Pre-treatment Score") +
    theme_bw() + theme(legend.position = "none", axis.title.x = element_blank())
  
  return(p)
}

# ==============================================================================
# SET 1: TISSUE Analysis
# ==============================================================================
# 选择 Tissue 中感兴趣的细胞 (例如: Pr_B, CD8_T, NK, Plasma)
cells_tissue <- c("TAMs", "Mast_Cells")
#cells_tissue = annotations_tissue %>% unlist

p1_tissue <- plot_mechanism(cbi_tissue, "Tissue (TME)")
p2_tissue <- plot_longitudinal(cbi_tissue, cells_tissue, "Tissue (TME)")
p3_tissue <- plot_baseline(cbi_tissue, cells_tissue, "Tissue (TME)")

#print(p1_tissue)
print(p2_tissue)
#print(p3_tissue)

# ==============================================================================
# SET 2: BLOOD Analysis
# ==============================================================================
# 选择 Blood 中感兴趣的细胞 (例如: Proliferating, NK, HSPC, Platelets)
cells_blood <- c("HSPC", 'Naive_B',"pDCs")
#cells_blood = annotations_blood %>% unlist()

p1_blood <- plot_mechanism(cbi_blood, "Blood (PBMC)")
p2_blood <- plot_longitudinal(cbi_blood, cells_blood, "Blood (PBMC)")
p3_blood <- plot_baseline(cbi_blood, cells_blood, "Blood (PBMC)")

print(p1_blood)
print(p2_blood)
print(p3_blood)
 
p2_tissue/p3_blood

tissue_exp = read_delim("/Users/hechang/Documents/chenlab/scRNA/tissue/Tissue_Pseudo_Bulk_by_Cluster_Corrected.csv")

# ==============================================================================
# 0. 准备工作：定义 M2 基因集
# ==============================================================================
# 经典的 M2 巨噬细胞 / 免疫抑制相关基因
m2_genes <- c("CD163", "MRC1", "MS4A4A", "STAB1", "TGFB1", "IL10", "FN1", "VSIG4", "MSR1")

# ==============================================================================
# 1. 处理表达矩阵 (tissue_exp)
# ==============================================================================
# 假设 tissue_exp 已经读取
# 格式: 行=基因, 列=SampleID (如 P002-Post-0)

# 1.1 转置矩阵 (变为 行=样本, 列=基因)
# 先把第一列变成行名
exp_mat <- tissue_exp %>%
  tibble::column_to_rownames("...1") %>% # 你的第一列叫 ...1
  as.matrix() %>%
  t() %>% 
  as.data.frame()

# 1.2 筛选 TAMs (Cluster 2)
# 根据列名 P002-Post-0，最后一个数字是 Cluster ID
# 我们只保留结尾是 "-2" 的行 (假设 Cluster 2 是 TAMs)
tam_exp <- exp_mat %>%
  tibble::rownames_to_column("Full_ID") %>%
  filter(grepl("-2$", Full_ID)) %>% # 筛选 Cluster 2
  tibble::column_to_rownames("Full_ID")

# 1.3 计算 M2 Score (使用 Z-score 平均法)
# 确保基因在矩阵中存在
valid_m2_genes <- intersect(m2_genes, colnames(tam_exp))
print(paste("Used M2 Genes:", paste(valid_m2_genes, collapse=", ")))

if(length(valid_m2_genes) > 0) {
  # 对表达矩阵进行 Z-score 标准化 (按列/基因)
  tam_scaled <- scale(tam_exp)
  
  # 计算 M2 基因的平均 Z-score
  tam_scores <- data.frame(
    SampleID_Raw = rownames(tam_exp),
    M2_Score = rowMeans(tam_scaled[, valid_m2_genes], na.rm = TRUE)
  )
} else {
  stop("No M2 genes found in expression matrix!")
}

# ==============================================================================
# 2. 合并 CBI 数据
# ==============================================================================
# 假设 scRNA_cbi 是你之前算好的 CBI 结果
# 我们需要构建一个匹配的 ID 键

# scRNA_cbi 里有 Patient, Status, Cluster
# tissue_exp 的列名格式是: Patient-Status-Cluster (例如 P002-Post-0)

tam_cbi_merged <- cbi_tissue %>%
  # 1. 筛选 TAMs
  filter(CellType == "TAMs") %>%
  # 2. 构建与表达矩阵一致的 ID
  mutate(SampleID_Raw = paste(Patient, Status, Cluster, sep = "-")) %>%
  # 3. 合并 M2 Score
  inner_join(tam_scores, by = "SampleID_Raw")

print(head(tam_cbi_merged))

model <- lm(M2_Score ~ CBI, data = tam_cbi_merged)

# 2. 计算 Cook's Distance
cooksd <- cooks.distance(model)

# 3. 定义离群值阈值 (常用标准: 4 / 样本量)
n <- nrow(tam_cbi_merged)
threshold <- 4 / n

# 4. 标记并过滤离群点
# 我们保留 Cook's D 小于阈值的点
clean_data <- tam_cbi_merged %>%
  mutate(Cooks_D = cooksd) %>%
  filter(Cooks_D < threshold) # 自动排除强影响点

# 查看排除了几个点
n_removed <- n - nrow(clean_data)
print(paste("自动排除了", n_removed, "个离群点"))

# 1. 分组
# 这里的 clean_data 是你去除离群点后的数据，或者用原始数据 tam_cbi_merged
plot_data_box <- clean_data %>%
  mutate(CBI_Group = ifelse(CBI > median(CBI, na.rm = TRUE), "High Score", "Low Score"))

# 设定顺序
plot_data_box$CBI_Group <- factor(plot_data_box$CBI_Group, levels = c("Low Score", "High Score"))

ggplot(plot_data_box, aes(x = CBI_Group, y = M2_Score, fill = CBI_Group)) +
  
  # 1. 添加小提琴图 (展示数据密度分布)
  geom_violin(
    trim = FALSE, # 设置为 FALSE 以延伸到数据的最大最小值
    alpha = 0.6,
    scale = "width" # 让小提琴的宽度反映样本量
  ) +
  
  # 2. 在小提琴图内添加一个简化的箱线图 (可选, 用于显示中位数和四分位数)
  geom_boxplot(
    width = 0.15, # 缩小箱线图的宽度
    outlier.shape = NA,
    fill = "white", # 使用白色填充箱线图
    alpha = 0.5 
  ) +
  
  # 3. 添加原始数据散点
  geom_jitter(width = 0.1, size = 2, alpha = 0.5) +
  
  # 4. 统计检验
  stat_compare_means(method = "wilcox.test",label.x = 1.3) + # 调整 p 值标签的垂直位置
  
  # 5. 颜色和标签 (保持不变)
  scale_fill_manual(values = c("Low Score" = "#4682B4", "High Score" = "#DC143C")) +
  
  labs(
    title = "High Score Indicates Immunosuppression (Violin Plot)",
    x = "Predicted Group",
    y = "M2 Signature Score"
  ) +
  
  # 6. 主题 (保持不变)
  theme_classic() +
  theme(legend.position = "none")



# ==============================================================================
# Panel D: Targeting Myeloid Checkpoints 
# ==============================================================================

# 假设你之前提取了 tam_exp (TAMs 表达矩阵)
target_genes <- c("CD274","LILRB4") # 或者 "CD274"
targeting_plots = list()

for (target_gene in target_genes){
  tam_checkpoints <- data.frame(
    SampleID_Raw = rownames(tam_exp),
    Expr = scale(tam_exp[, target_gene]) # Z-score
  )
  
  # 2. 合并 CBI (TAMs Level)
  plot_data_target <- cbi_tissue %>%
    filter(CellType == "TAMs") %>%
    mutate(SampleID_Raw = paste(Patient, Status, Cluster, sep = "-")) %>%
    inner_join(tam_checkpoints, by = "SampleID_Raw")
  
  model <- lm(Expr ~ CBI, data =plot_data_target)
  
  # 2. 计算 Cook's Distance
  cooksd <- cooks.distance(model)
  
  # 3. 定义离群值阈值 (常用标准: 4 / 样本量)
  n <- nrow(plot_data_target)
  threshold <- 4 / n
  
  # 4. 标记并过滤离群点
  # 我们保留 Cook's D 小于阈值的点
  clean_data <- plot_data_target %>%
    mutate(Cooks_D = cooksd) %>%
    filter(Cooks_D < threshold) # 自动排除强影响点
  
  # 查看排除了几个点
  n_removed <- n - nrow(clean_data)
  print(paste("自动排除了", n_removed, "个离群点"))
  
  
  # 3. 绘图
  p_d_new <- ggplot(clean_data, aes(x = CBI, y = Expr)) +
    geom_point(aes(color = Status), size = 3, alpha = 0.7) +
    geom_smooth(method = "lm", color = "black", linetype = "dashed") +
    stat_cor(method = "spearman", label.x.npc = "left") +
    scale_color_manual(values = c("Pre" = "#DC143C", "Post" = "#4682B4", "Prog" = "black")) +
    labs(
      x = "TAMs Response Score",
      y = paste0(target_gene, " Expression (Z-Score)")
    ) +
    theme_classic()
  
  targeting_plots[[target_gene]] = p_d_new
}
combined_plot <- (targeting_plots$CD274 + targeting_plots$LILRB4) +
  plot_layout(guides = "collect") + # <--- 收集所有图例到一个地方
  plot_annotation(
    title = 'Myeloid Immunotherapy Targets Correlation', # <--- 设置总标题
    subtitle = 'Correlation between TAMs Response Score and Target Gene Expression (Z-Score)',
    theme = theme_classic() # 可选：确保总标题使用相同的风格
  )

print(combined_plot)
