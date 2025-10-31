import pandas as pd
import torch
import torch.nn as nn
import os

# Load CSV once
# csv_path = "/mnt/home/at3836/Contrastive_Learning/data/final_data/ASCOT_finetuning/"
# df = pd.read_csv(csv_path)
# df.set_index("exon_id", inplace=True)
# tissue_columns = df.columns[10:-3]  # assumes 56 tissue PSI columns

class multitissue_MSE(nn.Module):
    def __init__(self, csv_dir):
        super().__init__()
        self.csv_dir = csv_dir
        self.cache = {}  # cache loaded DataFrames
        self.tissue_cols = None

    def _load_df(self, split):
        if split not in self.cache:
            path = os.path.join(self.csv_dir, f"{split}_cassette_exons_with_logit_mean_psi.csv")
            df = pd.read_csv(path)
            df.set_index("exon_id", inplace=True)
            self.cache[split] = df
            if self.tissue_cols is None:
                self.tissue_cols = df.columns[11:-3]
        return self.cache[split]

    def forward(self, delta_logits, exon_ids, split):
        """
        Args:
            delta_logits (Tensor): (B, 56) - model predicts Δlogit(Ψₑₜ)
            exon_ids (List[str]): length-B list of exon IDs

        Returns:
            Scalar loss over valid tissue PSI observations
        """

        device = delta_logits.device

        # Lookup ground-truth values from DataFrame
        df = self._load_df(split)
        df_batch = df.loc[exon_ids]

        # PSI true values (scaled to 0–1) and mask
        psi_true = torch.tensor(
            df_batch[self.tissue_cols].values / 100.0,
            dtype=torch.float32, device=device
        )
        psi_mask = ~torch.isnan(psi_true)
        eps = 1e-6
        # Replace NaNs with something safe just so logit doesn’t blow up
        psi_true_safe = torch.clamp(
            psi_true.nan_to_num(0.5),  # placeholder, won’t be used where mask=False
            eps, 1 - eps
        )

        logit_mean_psi = torch.tensor(
            df_batch["logit_mean_psi"].values,
            dtype=torch.float32, device=device
        )

        delta_logits_true = torch.logit(psi_true_safe) - logit_mean_psi[:, None]

        loss_el = nn.MSELoss(reduction='none')(delta_logits, delta_logits_true)
        masked_loss = loss_el * psi_mask.float()         # zero out invalid entries
        loss = masked_loss.sum() / psi_mask.sum().clamp_min(1)
        return loss







        # masked_loss = loss * psi_mask   


        # # psi_pred = torch.sigmoid(logits)

        # # Compute KL divergence terms
        # eps = 1e-7
        # term1 = psi_true * torch.log((psi_true + eps) / (psi_pred + eps))
        # term2 = (1 - psi_true) * torch.log((1 - psi_true + eps) / (1 - psi_pred + eps))
        # kl = term1 + term2

        # # Apply mask and average over observed values
        # masked_kl = kl * psi_mask
        # loss = masked_kl.sum() / psi_mask.sum().clamp(min=1)

        # # Add logit(mean_psi) back to delta and apply sigmoid
        # logit_mean_psi = torch.tensor(
        #     df_batch["logit_mean_psi"].values,
        #     dtype=torch.float32, device=device
        # )
        # logits = delta_logits + logit_mean_psi[:, None]
        # psi_pred = torch.sigmoid(logits)

        # # Compute KL divergence terms
        # eps = 1e-7
        # term1 = psi_true * torch.log((psi_true + eps) / (psi_pred + eps))
        # term2 = (1 - psi_true) * torch.log((1 - psi_true + eps) / (1 - psi_pred + eps))
        # kl = term1 + term2

        # # Apply mask and average over observed values
        # masked_kl = kl * psi_mask
        # loss = masked_kl.sum() / psi_mask.sum().clamp(min=1)

        return loss
    

        