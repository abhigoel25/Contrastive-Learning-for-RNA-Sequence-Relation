import torch
import torch.nn as nn
import lightning.pytorch as pl
from hydra.utils import instantiate
from torchmetrics import R2Score
import time
from scipy.stats import spearmanr
from scipy.special import expit, logit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


# --------------------------
# 1. Load and prepare ground truth
# --------------------------
def load_ground_truth(gt_path: str, pred_shape_M: int) -> tuple[pd.DataFrame, list[str]]:
    ground_truth = pd.read_csv(gt_path)
    cols = list(ground_truth.columns)

    # Find tissue columns
    start_idx = cols.index('exon_boundary') + 1 if 'exon_boundary' in cols else 0
    if 'chromosome' in cols:
        end_idx = cols.index('chromosome')
    elif 'mean_psi' in cols:
        end_idx = cols.index('mean_psi')
    else:
        end_idx = cols.index('logit_mean_psi')
    tissue_cols = cols[start_idx:end_idx]

    # Adjust count if mismatch
    if len(tissue_cols) != pred_shape_M:
        tissue_cols = tissue_cols[:pred_shape_M]
        print(f"[warn] GT tissues ({len(tissue_cols)}) != pred dims ({pred_shape_M}); truncating.")

    gt = ground_truth[['exon_id', 'logit_mean_psi'] + tissue_cols].copy()
    return gt, tissue_cols

# --------------------------
# 2. Merge predictions with ground truth
# --------------------------
def merge_predictions(gt: pd.DataFrame, exon_ids, y_pred, tissue_cols: list[str]) -> pd.DataFrame:
    pred_delta_cols = [f"{t}_pred_delta" for t in tissue_cols]
    pred_df = pd.DataFrame(y_pred, columns=pred_delta_cols)
    pred_df.insert(0, 'exon_id', exon_ids)
    merged = gt.merge(pred_df, on='exon_id')
    return merged


# --------------------------
# 3. Compute final PSI predictions (in %)
# --------------------------
def compute_final_psi(merged: pd.DataFrame, tissue_cols: list[str]) -> pd.DataFrame:
    logit_mean = merged['logit_mean_psi'].to_numpy()[:, None]  # (N,1)
    pred_delta_mat = merged[[f"{t}_pred_delta" for t in tissue_cols]].to_numpy()
    
    
    final_psi_pct = 100.0 * expit(pred_delta_mat + logit_mean)  # (N,M)
    pred_psi_df = pd.DataFrame(final_psi_pct, columns=tissue_cols)
    pred_psi_df.insert(0, 'exon_id', merged['exon_id'].values)
    return pred_psi_df


# --------------------------
# 4. Per-tissue correlations
# --------------------------
def compute_per_tissue_corr(merged: pd.DataFrame, pred_psi_df: pd.DataFrame, tissue_cols: list[str]) -> pd.DataFrame:
    eps = 1e-6
    rows = []

    for t in tissue_cols:
        truth_psi_pct = merged[t]
        pred_psi_pct_t = pred_psi_df[t]
        pred_delta_t = merged[f"{t}_pred_delta"]

        valid = (
            truth_psi_pct.between(0, 100) &
            truth_psi_pct.notna() &
            merged['logit_mean_psi'].notna() &
            pred_delta_t.notna() &
            pred_psi_pct_t.notna()
        )

        if valid.any():
            rho_psi, _ = spearmanr(truth_psi_pct.loc[valid], pred_psi_pct_t.loc[valid])
        else:
            rho_psi = np.nan

        truth_frac = np.clip(truth_psi_pct / 100.0, eps, 1 - eps)
        truth_delta = pd.Series(logit(truth_frac) - merged['logit_mean_psi'].to_numpy(), index=merged.index)

        if valid.any():
            rho_delta, _ = spearmanr(truth_delta.loc[valid], pred_delta_t.loc[valid])
        else:
            rho_delta = np.nan

        rows.append({
            'tissue': t,
            'spearman_psi': float(rho_psi) if rho_psi == rho_psi else np.nan,
            'spearman_delta': float(rho_delta) if rho_delta == rho_delta else np.nan,
            'n_valid_psi': int(valid.sum()),
            'n_valid_delta': int(valid.sum()),
        })

    return pd.DataFrame(rows).sort_values('tissue').reset_index(drop=True)


# --------------------------
# 5. Print + save results
# --------------------------
def save_and_report(out_dir: Path, pred_psi_df: pd.DataFrame, metrics_df: pd.DataFrame):
    pred_out_path = out_dir / "tsplice_final_predictions_all_tissues.tsv"
    metrics_out_path = out_dir / "tsplice_spearman_by_tissue.tsv"

    pred_psi_df.to_csv(pred_out_path, sep="\t", index=False)
    metrics_df.to_csv(metrics_out_path, sep="\t", index=False)

    print("\n=== Per-tissue Spearman correlations ===")
    for _, r in metrics_df.iterrows():
        print(f"{r['tissue']}: PSI={r['spearman_psi']:.4f} | Δlogit={r['spearman_delta']:.4f} "
              f"(nψ={r['n_valid_psi']}, nΔ={r['n_valid_delta']})")

    print(f"\nSaved predictions to: {pred_out_path}")
    print(f"Saved per-tissue metrics to: {metrics_out_path}")
    print(pred_psi_df.head())


def resolve_out_dir(ckpt_dir_str: str) -> Path:
    """
    If the checkpoint dir looks like:
      .../Contrastive_Learning/files/results/exprmnt_YYYY_MM_DD__HH_MM_SS/weights/checkpoints
    -> save to:
      .../Contrastive_Learning/files/output_files
    Else -> save to the checkpoint dir itself.
    """
    ckpt_dir = Path(ckpt_dir_str)
    s = str(ckpt_dir)

    looks_like_exprmnt = (
        "/Contrastive_Learning/files/results/exprmnt_" in s
        and ckpt_dir.name == "checkpoints"
        and ckpt_dir.parent.name == "weights"
    )
    if looks_like_exprmnt:
        # walk up to the "Contrastive_Learning" anchor, then use files/output_files
        p = ckpt_dir
        while p.name != "Contrastive_Learning" and p != p.parent:
            p = p.parent
        return p / "files" / "output_files"
    else:
        return ckpt_dir
    


class MTSpliceBCE(pl.LightningModule):
    def __init__(self, encoder, config, embed_dim=32, out_dim=56, dropout=0.5):
        super().__init__()
        # self.save_hyperparameters(ignore=['encoder'])

        self.encoder = encoder
        self.config = config

        if self.config.aux_models.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False


        if hasattr(encoder, "get_last_embedding_dimension") and callable(encoder.get_last_embedding_dimension):
            print("📏 Using encoder.get_last_embedding_dimension()")
            encoder_output_dim = encoder.get_last_embedding_dimension()

        else:
            print("⚠️ Warning: `encoder.output_dim` not defined, falling back to dummy input.")
            if hasattr(config, "dataset") and hasattr(config.dataset, "seq_len"):
                seq_len = config.dataset.seq_len
            else:
                raise ValueError("`seq_len` not found in config.dataset — can't create dummy input.")

            dummy_input = torch.full((1, 4, seq_len), 1.0)  # one-hot-style dummy input
            dummy_input = dummy_input.to(next(encoder.parameters()).device)

            with torch.no_grad():
                dummy_output = encoder(dummy_input)
                encoder_output_dim = dummy_output.shape[-1]

            print(f"📏 Inferred encoder output_dim = {encoder_output_dim}")
        
        self.fc1 = nn.Linear(encoder_output_dim, embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_dim, out_dim)


        # Instantiate loss and metrics via Hydra
        self.loss_fn = instantiate(config.loss)

        self.metric_fns = []
        for metric in config.task.metrics:
            if metric == "r2_score":
                self.metric_fns.append(R2Score())

    
    def forward(self, x):
        
        seql, seqr = x
        features = self.encoder(seql, seqr)
        x = self.fc1(features)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x    
        

    def training_step(self, batch, batch_idx):
        x, y, exon_ids = batch
        y_pred = self(x).squeeze()
        
        # if self.config.aux_models.mtsplice_BCE:
        try:
            loss_term = self.config.loss['_target_'].split('.')[-1]
        except:
            loss_term = None
        if loss_term == 'MTSpliceBCELoss' or loss_term == 'multitissue_MSE':
            loss = self.loss_fn(y_pred, exon_ids, split='train')
        else:
            loss = self.loss_fn(y_pred, y)

        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        for metric_fn in self.metric_fns:
            if self.config.aux_models.mtsplice_BCE:
                break
            self.log(f"train_{metric_fn.__class__.__name__}", metric_fn(y_pred, y), on_epoch=True, prog_bar=True, sync_dist=True)

        # DO NOT ERASE (AT)
        # --- GPU memory logging every N batches ---
        if batch_idx % 5 == 0 and torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved  = torch.cuda.memory_reserved(0) / (1024**3)
            peak_alloc = torch.cuda.max_memory_allocated(0) / (1024**3)
            peak_reserved = torch.cuda.max_memory_reserved(0) / (1024**3)
            print(f"[Batch {batch_idx}] "
                f"Allocated: {allocated:.2f} GiB | "
                f"Reserved: {reserved:.2f} GiB | "
                f"PeakAlloc: {peak_alloc:.2f} GiB | "
                f"PeakReserved: {peak_reserved:.2f} GiB")
        return loss

    def validation_step(self, batch, batch_idx):
        
        x, y, exon_ids = batch
        y_pred = self(x).squeeze()
        
        if self.config.aux_models.mtsplice_BCE:
            loss = self.loss_fn(y_pred, exon_ids, split='val')
        else:
            loss = self.loss_fn(y_pred, y)

        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        for metric_fn in self.metric_fns:
            if self.config.aux_models.mtsplice_BCE:
                break
            self.log(f"val_{metric_fn.__class__.__name__}", metric_fn(y_pred, y), on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    
    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_targets = []
        self.test_exon_ids = []

    def test_step(self, batch, batch_idx):
        x, y, exon_ids = batch
        y_pred = self(x).squeeze()
        
        if self.config.aux_models.mtsplice_BCE:
            split = self.config.dataset.test_files.intronexon.split('/')[-1].split('_')[1]
            loss = self.loss_fn(y_pred, exon_ids, split)
        else:
            loss = self.loss_fn(y_pred, y)

        self.log("test_loss", loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        for metric_fn in self.metric_fns:
            if self.config.aux_models.mtsplice_BCE:
                break
            self.log(f"test_{metric_fn.__class__.__name__}", metric_fn(y_pred, y), on_epoch=True, prog_bar=True, sync_dist=True)

        # Store for Spearman
        self.test_preds.append(y_pred.detach().cpu())
        self.test_targets.append(y.detach().cpu())

        # === NEW: Store exon IDs
        self.test_exon_ids += list(exon_ids)

        return loss
    
    

    def on_test_epoch_end(self):
        from scipy.special import logit, expit
        from scipy.stats import spearmanr
        import numpy as np
        import pandas as pd

        # ---- Where to save
        out_dir = resolve_out_dir(self.config.callbacks.model_checkpoint.dirpath)
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.config.aux_models.mtsplice_BCE:

            # # Predictions
            y_pred = torch.cat(self.test_preds, dim=0).cpu().numpy()     # shape: [N, M]
            exon_ids = list(self.test_exon_ids)                           # length N
            N, M = y_pred.shape

            # y_pred: (N, M), exon_ids: list of length N
            split = self.config.dataset.test_files.intronexon.split('/')[-1].split('_')[1]
            gt_path = f"{self.config.loss.csv_dir}/{split}_cassette_exons_with_logit_mean_psi.csv"
            gt, tissue_cols = load_ground_truth(gt_path, y_pred.shape[1])
            merged = merge_predictions(gt, exon_ids, y_pred, tissue_cols)

            pred_out_path = out_dir / "tsplice_raw_output_all_tissues.tsv"
            merged.to_csv(pred_out_path, sep="\t", index=False)
            
            pred_psi_df = compute_final_psi(merged, tissue_cols)
            metrics_df = compute_per_tissue_corr(merged, pred_psi_df, tissue_cols)
            save_and_report(out_dir, pred_psi_df, metrics_df)

            # log averages if needed
            tissue_name = 'Retina - Eye'
            psi_value = metrics_df.loc[metrics_df["tissue"] == tissue_name, "spearman_psi"].iloc[0]
            psi_delta = metrics_df.loc[metrics_df["tissue"] == tissue_name, "spearman_delta"].iloc[0]
            self.log(f"{tissue_name}_spearman_psi", psi_value, prog_bar=True, sync_dist=True)
            self.log(f"{tissue_name}_spearman_delta", psi_delta, prog_bar=True, sync_dist=True)

            self.log("test_mean_spearman_psi", metrics_df['spearman_psi'].mean(skipna=True), prog_bar=True, sync_dist=True)
            self.log("test_mean_spearman_delta", metrics_df['spearman_delta'].mean(skipna=True), prog_bar=True, sync_dist=True)

        else:
            # ==============================
            # Simple regression path (no per-tissue outputs)
            # ==============================
            y_pred_all = torch.cat(self.test_preds).cpu().numpy()
            y_true_all = torch.cat(self.test_targets).cpu().numpy()

            eps = 1e-6
            from scipy.special import logit
            y_true_logit = logit(np.clip(y_true_all / 100.0, eps, 1 - eps))
            y_pred_logit = logit(np.clip(y_pred_all / 100.0, eps, 1 - eps))

            from scipy.stats import spearmanr
            rho, _ = spearmanr(y_true_logit, y_pred_logit)
            self.log("test_spearman_logit", rho, prog_bar=True, sync_dist=True)
            print(f"\n🔬 Spearman ρ (logit PSI, test set): {rho:.4f}")

            df = pd.DataFrame({
                "index": np.arange(len(y_true_all)),
                "y_true": y_true_all,
                "y_pred": y_pred_all,
                "y_true_logit": y_true_logit,
                "y_pred_logit": y_pred_logit,
            })

            pred_out_path = out_dir / "psi_regression_test_predictions.tsv"
            df.to_csv(pred_out_path, sep="\t", index=False)
            print(f"Saved to: {pred_out_path}")
            print(df.head())


    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        gpu_memory = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
        reserved_memory = torch.cuda.memory_reserved(0) / 1e9 if torch.cuda.is_available() else 0
        peak_memory = torch.cuda.max_memory_reserved(0) / 1e9 if torch.cuda.is_available() else 0

        self.log("epoch_time", epoch_time, prog_bar=True, sync_dist=True)
        self.log("gpu_memory_usage", gpu_memory, prog_bar=True, sync_dist=True)
        self.log("gpu_reserved_memory", reserved_memory, prog_bar=True, sync_dist=True)
        self.log("gpu_peak_memory", peak_memory, prog_bar=True, sync_dist=True)

        print(f"\nEpoch {self.current_epoch} took {epoch_time:.2f} seconds.")
        print(f"GPU Memory Used: {gpu_memory:.2f} GB, Reserved: {reserved_memory:.2f} GB, Peak: {peak_memory:.2f} GB")

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            print(f"Setting up training for {self.config.task._name_}")
        if stage == 'validate' or stage is None:
            print(f"Setting up validation for {self.config.task._name_}")

    def configure_optimizers(self):
        return instantiate(self.config.optimizer, params=self.parameters())
    



    