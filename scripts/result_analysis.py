import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import time

trimester = time.strftime("_%Y_%m_%d__%H_%M_%S")


# 1. Read the CSV
# df = pd.read_csv("/gpfs/commons/home/atalukder/Contrastive_Learning/files/test_predictions_with_index__2025_06_23__18_13_18.csv")
# df = pd.read_csv("/gpfs/commons/home/atalukder/Contrastive_Learning/files/VariableExon_predictions_with_index.csv")
# df = pd.read_csv("/gpfs/commons/home/atalukder/Contrastive_Learning/files/test_predictions_with_index__2025_06_24__12_31_53.csv")

df = pd.read_csv("/gpfs/commons/home/atalukder/Contrastive_Learning/files/test_predictions_with_index__2025_06_24__12_38_34.csv")

# # Filter valid PSI values
# df = df[(df["y_true"] >= 0) & (df["y_true"] <= 100)]
# df = df[(df["y_pred"] >= 0) & (df["y_pred"] <= 100)]

# # Plot histograms
# plt.figure(figsize=(8, 5))
# sns.histplot(df["y_true"], label="y_true", color="skyblue", bins=30, stat="density", alpha=0.5)
# sns.histplot(df["y_pred"], label="y_pred", color="orange", bins=30, stat="density", alpha=0.5)

# plt.title("Histogram of PSI: y_true vs y_pred")
# plt.xlabel("PSI Value")
# plt.ylabel("Density")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f"/gpfs/commons/home/atalukder/Contrastive_Learning/code/ML_model/figures/psi_histogram_{trimester}.png")
# plt.show()


# # Filter out outliers in y_pred
# filtered_df = df[df["y_pred"] <= 100]

# # Scatter plot
# plt.figure(figsize=(8, 6))
# sns.scatterplot(data=filtered_df, x="y_true", y="y_pred", alpha=0.6)
# plt.title("Filtered Scatter Plot: y_true vs y_pred (y_pred ≤ 100)")
# plt.xlabel("y_true (PSI)")
# plt.ylabel("y_pred (PSI)")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f"/gpfs/commons/home/atalukder/Contrastive_Learning/code/ML_model/figures/filtered_y_true_vs_y_pred_{trimester}.png")
# plt.show()


# # 4. Scatter plot: y_true vs y_pred
# plt.figure(figsize=(8, 6))
# sns.scatterplot(data=df, x="y_true", y="y_pred", alpha=0.6)
# plt.title("Scatter Plot of y_true vs y_pred")
# plt.xlabel("y_true (PSI)")
# plt.ylabel("y_pred (PSI)")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f"/gpfs/commons/home/atalukder/Contrastive_Learning/code/ML_model/figures/y_true_vs_y_pred_{trimester}.png")
# plt.show()

# 5. Pearson correlation
r, p = pearsonr(df["y_true"], df["y_pred"])
print(f"\n🔬 Pearson Correlation (y_true vs y_pred): r = {r:.4f}, p-value = {p:.4e}")
