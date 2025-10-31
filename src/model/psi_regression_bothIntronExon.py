import torch
import torch.nn as nn
import lightning.pytorch as pl
from hydra.utils import instantiate
from torchmetrics import R2Score
import time
from scipy.stats import spearmanr, pearsonr
from scipy.special import logit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class PSIRegressionModel(pl.LightningModule):
    def __init__(self, encoder_5p, encoder_3p, encoder_exon, config):
        super().__init__()
        # self.save_hyperparameters(ignore=['encoder'])

        self.encoder_5p = encoder_5p
        self.encoder_3p = encoder_3p
        self.config = config
        self.mode = config.aux_models.mode

        if self.config.aux_models.freeze_encoder:
            for param in self.encoder_5p.parameters():
                param.requires_grad = False
            for param in self.encoder_3p.parameters():
                param.requires_grad = False

       
        if hasattr(encoder_5p, "get_last_embedding_dimension") and callable(encoder_5p.get_last_embedding_dimension):
            print("📏 Using encoder.get_last_embedding_dimension()")
            encoder_output_dim = encoder_5p.get_last_embedding_dimension()

        else:
            print("⚠️ Warning: `encoder.output_dim` not defined, falling back to dummy input.")
            if hasattr(config, "dataset") and hasattr(config.dataset, "seq_len"):
                seq_len = config.dataset.seq_len
            else:
                raise ValueError("`seq_len` not found in config.dataset — can't create dummy input.")

            dummy_input = torch.full((1, 4, seq_len), 1.0)  # one-hot-style dummy input
            dummy_input = dummy_input.to(next(encoder_5p.parameters()).device)

            with torch.no_grad():
                dummy_output = encoder_5p(dummy_input)
                encoder_output_dim = dummy_output.shape[-1]

            print(f"📏 Inferred encoder output_dim = {encoder_output_dim}")
        
        if self.mode == "intronOnly":
            total_dim = encoder_output_dim * 2  # concat of 3 encoders
        else:
            self.encoder_exon = encoder_exon
            total_dim = encoder_output_dim * 3  # con

        self.regressor = nn.Sequential(
            nn.Linear(total_dim, config.aux_models.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.aux_models.hidden_dim, config.aux_models.output_dim))


        # Instantiate loss and metrics via Hydra
        self.loss_fn = instantiate(config.loss)

        self.metric_fns = []
        for metric in config.task.metrics:
            if metric == "r2_score":
                self.metric_fns.append(R2Score())

    
    def forward(self, x):
        if self.mode == "intronOnly":
            x_5p, x_3p, _ = x
            emb_5p = self.encoder_5p(x_5p)
            emb_3p = self.encoder_3p(x_3p)
            features = torch.cat([emb_5p, emb_3p], dim=-1)

            return self.regressor(features)

        else:
            x_5p, x_3p, x_exon = x
            emb_5p = self.encoder_5p(x_5p)
            emb_3p = self.encoder_3p(x_3p)
            emb_exon = self.encoder_exon(x_exon)

            features = torch.cat([emb_5p, emb_exon, emb_3p], dim=-1)

            return self.regressor(features)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_pred = self(x).squeeze()
        # y_pred = 100 * torch.sigmoid(y_pred)
        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
            print(f"\n[🚨 Warning] NaN or Inf in y_pred at batch {batch_idx}")
            print(f"y_pred: {y_pred}")

        # y_pred = self(x)

        # # Combine masks for any invalid predictions or targets
        # valid_mask = ~(torch.isnan(y_pred) | torch.isinf(y_pred) | torch.isnan(y) | torch.isinf(y))
        # # Filter out invalid values
        # y_pred = y_pred[valid_mask]
        # y = y[valid_mask]
        
        loss = self.loss_fn(y_pred, y)
        
        
        # print(f"batch {batch_idx}, loss {loss}")
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        for metric_fn in self.metric_fns:
            # value = metric_fn(y_pred, y)
            # print(f"🔍 Metric ({metric_fn.__class__.__name__}): {value.item()}")
            self.log(f"train_{metric_fn.__class__.__name__}", metric_fn(y_pred, y), on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        # y_pred = self(x)
        y_pred = self(x).squeeze()
        # y_pred = 100 * torch.sigmoid(y_pred)

        # Combine masks for any invalid predictions or targets
        # valid_mask = ~(torch.isnan(y_pred) | torch.isinf(y_pred) | torch.isnan(y) | torch.isinf(y))
        # # Filter out invalid values
        # y_pred = y_pred[valid_mask]
        # y = y[valid_mask]
        loss = self.loss_fn(y_pred, y)

        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        for metric_fn in self.metric_fns:
            self.log(f"val_{metric_fn.__class__.__name__}", metric_fn(y_pred, y), on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, exon_ids = batch
        y_pred = self(x).squeeze()
        # y_pred = 100 * torch.sigmoid(y_pred)
        loss = self.loss_fn(y_pred, y)

        for metric_fn in self.metric_fns:
            self.log(f"test_{metric_fn.__class__.__name__}", metric_fn(y_pred, y), on_epoch=True, prog_bar=True, sync_dist=True)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # Store for Spearman
        self.test_preds.append(y_pred.detach().cpu())
        self.test_targets.append(y.detach().cpu())

        # === NEW: Store exon IDs
        self.test_exon_ids += list(exon_ids)

        return loss

    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_targets = []
        self.test_exon_ids = []

    def on_test_epoch_end(self):
        y_pred_all = torch.cat(self.test_preds).numpy()
        y_true_all = torch.cat(self.test_targets).numpy()

        # rho, _ = spearmanr(y_true_all, y_pred_all)
        # self.log("test_spearman", rho, prog_bar=True, sync_dist=True)
        # print(f"\n🔬 Spearman ρ (test set): {rho:.4f}")

        # Apply logit transformation with clamping to avoid log(0)
        eps = 1e-6
        y_true_logit = logit(np.clip(y_true_all/100, eps, 1 - eps))
        y_pred_logit = logit(np.clip(y_pred_all/100, eps, 1 - eps))

        rho, _ = spearmanr(y_true_logit, y_pred_logit)
        self.log("test_spearman_logit", rho, prog_bar=True, sync_dist=True)
        print(f"\n🔬 Spearman ρ (logit PSI, test set): {rho:.4f}")

        pearson, _ = pearsonr(y_true_logit, y_pred_logit)
        self.log("test_pearson_logit", pearson, prog_bar=True, sync_dist=True)
        print(f"\n🔬 Pearson ρ (logit PSI, test set): {pearson:.4f}")

        import time
        trimester = time.strftime("_%Y_%m_%d__%H_%M_%S")

        # Save results to CSV
        df = pd.DataFrame({
            "index": np.arange(len(y_true_all)),
            "y_true": y_true_all,
            "y_pred": y_pred_all,
            "y_true_logit": y_true_logit,
            "y_pred_logit": y_pred_logit,
        })
        df.to_csv(f"/mnt/home/at3836/Contrastive_Learning/files/test_predictions_with_index_{trimester}.csv", index=False)



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
    



    