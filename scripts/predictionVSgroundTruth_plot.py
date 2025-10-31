import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logit
from typing import Tuple
import time

trimester = time.strftime("_%Y_%m_%d__%H_%M_%S")

def load_predictions(path: str) -> pd.DataFrame:
    """Load predictions TSV file."""
    df = pd.read_csv(path, sep="\t")
    df.rename(columns={df.columns[0]: "exon_id"}, inplace=True)
    return df

def load_ground_truth(path: str) -> pd.DataFrame:
    """Load ground truth CSV file."""
    df = pd.read_csv(path)
    df.rename(columns={df.columns[0]: "exon_id"}, inplace=True)
    return df

# def merge_data(gt_df: pd.DataFrame, pred_df: pd.DataFrame, tissue: str) -> Tuple[pd.Series, pd.Series]:
#     """
#     Merge ground truth and prediction data for a specific tissue.
#     Returns (y_true, y_pred).
#     """
#     if tissue not in gt_df.columns or tissue not in pred_df.columns:
#         raise ValueError(f"Tissue '{tissue}' not found in both files.")

#     merged = pd.merge(gt_df[["exon_id", tissue]], 
#                       pred_df[["exon_id", tissue]], 
#                       on="exon_id", 
#                       suffixes=("_true", "_pred"))

#     y_true = merged[f"{tissue}_true"].values
#     y_pred = merged[f"{tissue}_pred"].values

#     mask = ~pd.isna(y_true) & ~pd.isna(y_pred)
#     return y_true[mask], y_pred[mask]

# def plot_histogram(y_true, y_pred, tissue: str, save_path: str = None):
#     """Plot histogram of ground truth vs predictions for a tissue."""
#     plt.figure(figsize=(12, 5))
#     plt.hist(y_true, bins=50, alpha=0.5, label="y_true", density=True, color="skyblue")
#     plt.hist(y_pred, bins=50, alpha=0.5, label="y_pred", density=True, color="orange")

#     plt.xlabel("PSI Value")
#     plt.ylabel("Density")
#     plt.title(f"Histogram of PSI in {tissue}: y_true vs y_pred")
#     plt.legend()
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=300)  # save high-res figure
#         plt.close()
#     else:
#         plt.show()

# # -------------------------
# # Optional class wrapper
# # -------------------------
# class PSIPlotter:
#     def __init__(self, pred_path: str, gt_path: str):
#         self.pred_df = load_predictions(pred_path)
#         self.gt_df = load_ground_truth(gt_path)

#     def plot_for_tissue(self, tissue: str, save_path: str = None):
#         y_true, y_pred = merge_data(self.gt_df, self.pred_df, tissue)
#         plot_histogram(y_true, y_pred, tissue, save_path)

#     def plot_for_multiple_tissues(self, tissues: list):
#         for tissue in tissues:
#             try:
#                 self.plot_for_tissue(tissue)
#             except ValueError as e:
#                 print(e)

def merge_psi(gt_df: pd.DataFrame, pred_df: pd.DataFrame, tissue: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ground truth and predicted PSI values for a tissue."""
    if tissue not in gt_df.columns or tissue not in pred_df.columns:
        raise ValueError(f"Tissue '{tissue}' not found in both files.")
    merged = pd.merge(gt_df[["exon_id", tissue]], 
                      pred_df[["exon_id", tissue]], 
                      on="exon_id", 
                      suffixes=("_true", "_pred"))
    y_true = merged[f"{tissue}_true"].values
    y_pred = merged[f"{tissue}_pred"].values
    mask = ~pd.isna(y_true) & ~pd.isna(y_pred)
    return y_true[mask], y_pred[mask]


# def merge_delta_logit(gt_df: pd.DataFrame, pred_df: pd.DataFrame, tissue: str) -> Tuple[np.ndarray, np.ndarray]:
#     """Extract ground truth and predicted Δlogit values for a tissue."""
#     if tissue not in gt_df.columns or tissue not in pred_df.columns:
#         raise ValueError(f"Tissue '{tissue}' not found in both files.")
#     merged = pd.merge(
#         gt_df[["exon_id", "logit_mean_psi", tissue]],
#         pred_df[["exon_id", tissue]],
#         on="exon_id",
#         suffixes=("_gt", "_pred_delta")
#     )
#     eps = 1e-6
#     truth_frac = np.clip(merged[tissue] / 100.0, eps, 1 - eps)
#     delta_true = logit(truth_frac) - merged["logit_mean_psi"].to_numpy()
#     delta_pred = merged[f"{tissue}_pred_delta"].to_numpy()
#     mask = ~pd.isna(delta_true) & ~pd.isna(delta_pred)
#     return delta_true[mask], delta_pred[mask]

def compute_delta_logit(psi_percent: pd.Series, logit_mean: pd.Series) -> np.ndarray:
    """
    Convert PSI values (%) into Δlogit given logit_mean.
    Δlogit = logit(PSI/100) - logit_mean
    """
    from scipy.special import logit
    import numpy as np

    eps = 1e-6
    frac = np.clip(psi_percent / 100.0, eps, 1 - eps)
    return logit(frac) - logit_mean.to_numpy()


def merge_delta_logit(gt_df: pd.DataFrame, pred_df: pd.DataFrame, tissue: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract ground truth and predicted Δlogit values for a tissue.
    """
    if tissue not in gt_df.columns or tissue not in pred_df.columns:
        raise ValueError(f"Tissue '{tissue}' not found in both files.")

    merged = pd.merge(
        gt_df[["exon_id", "logit_mean_psi", tissue]],
        pred_df[["exon_id", tissue]],
        on="exon_id",
        suffixes=("_gt", "_pred")
    )

    # Ground truth Δlogit from PSI (%)
    delta_true = compute_delta_logit(merged[f"{tissue}_gt"], merged["logit_mean_psi"])

    # Predicted Δlogit (already Δlogit values, not PSI)
    # delta_pred = merged[f"{tissue}_pred"].to_numpy()
    delta_pred = compute_delta_logit(merged[f"{tissue}_pred"], merged["logit_mean_psi"])

    # Drop NaNs
    mask = ~pd.isna(delta_true) & ~pd.isna(delta_pred)
    return delta_true[mask], delta_pred[mask]

def plot_histogram(y_true, y_pred, tissue: str, value_type: str = "PSI", save_path: str = None):
    """Plot histogram for PSI or Δlogit depending on value_type."""
    plt.figure(figsize=(12, 5))
    plt.hist(y_true, bins=50, alpha=0.5, label=f"{value_type}_true", density=True, color="skyblue")
    plt.hist(y_pred, bins=50, alpha=0.5, label=f"{value_type}_pred", density=True, color="orange")

    plt.xlabel(value_type)
    plt.ylabel("Density")
    plt.title(f"Histogram of {value_type} in {tissue}: true vs predicted")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

# -------------------------
# Unified class
# -------------------------
class PSIPlotter:
    def __init__(self, pred_path: str, gt_path: str):
        self.pred_df = load_predictions(pred_path)
        self.gt_df = load_ground_truth(gt_path)

    def plot_psi(self, tissue: str, save_path: str = None):
        y_true, y_pred = merge_psi(self.gt_df, self.pred_df, tissue)
        plot_histogram(y_true, y_pred, tissue, value_type="PSI", save_path=save_path)

    def plot_delta_logit(self, tissue: str, save_path: str = None):
        delta_true, delta_pred = merge_delta_logit(self.gt_df, self.pred_df, tissue)
        plot_histogram(delta_true, delta_pred, tissue, value_type="Δlogit", save_path=save_path)

    def plot_for_multiple_tissues(self, tissues: list, value_type: str = "PSI"):
        for tissue in tissues:
            try:
                if value_type == "PSI":
                    self.plot_psi(tissue)
                elif value_type.lower() in ["delta", "Δlogit", "delta_logit"]:
                    self.plot_delta_logit(tissue)
                else:
                    raise ValueError(f"Unsupported value_type: {value_type}")
            except ValueError as e:
                print(e)


running_platform = 'NYGC'
prediction_file = "exprmnt_2025_08_27__17_11_29"
ground_truth_file = "variable_cassette_exons_with_logit_mean_psi.csv"
run_num = 1
tissue = "Retina - Eye"
# tissue = "Coronary - Artery"

if running_platform == 'NYGC':
    server_name = 'NYGC'
    server_path = '/gpfs/commons/home/atalukder/'
elif running_platform == 'EMPRAI':
    server_name = 'EMPRAI'
    server_path = '/mnt/home/at3836/'




prediction_final_path = f"{server_path}Contrastive_Learning/files/results/{prediction_file}/weights/checkpoints/run_{run_num}/tsplice_final_predictions_all_tissues.tsv"
ground_truth_final_path = f"{server_path}Contrastive_Learning/data/final_data/ASCOT_finetuning/{ground_truth_file}"
save_path_psi = f"{server_path}Contrastive_Learning/code/ML_model/scripts/figures/{tissue}_PSI_Histogram_{trimester}.png"
save_path_delta = f"{server_path}Contrastive_Learning/code/ML_model/scripts/figures/{tissue}_Delta_Logit_Histogram_{trimester}.png"


# # Initialize once
# plotter = PSIPlotter(prediction_final_path, ground_truth_final_path)

# # Plot for a single tissue
# plotter.plot_for_tissue(tissue, save_path=save_path)

# Plot for multiple tissues
# plotter.plot_for_multiple_tissues(["Retina - Eye", "Liver", "Lung"])

plotter = PSIPlotter(prediction_final_path, ground_truth_final_path)

# Plot PSI
# plotter.plot_psi(tissue, save_path=save_path_psi)

# Plot Δlogit
plotter.plot_delta_logit(tissue, save_path=save_path_delta)

# Plot multiple tissues (PSI or Δlogit)
# plotter.plot_for_multiple_tissues(["Retina - Eye", "Liver"], value_type="PSI")
# plotter.plot_for_multiple_tissues(["Retina - Eye", "Liver"], value_type="Δlogit")
