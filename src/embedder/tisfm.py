# src/embedder/tisfm_encoder.py

import os, sys
import torch.nn as nn
import torch

base_dir = os.path.dirname(__file__)
ata_root = os.path.join(base_dir, "tisfm_original", "ATAConv-main")

if ata_root not in sys.path:
    sys.path.insert(0, ata_root)

from models.model_pos_attention_calib_sigmoid_interaction import TISFM as RawTISFM


class TISFMEncoder(nn.Module):
    def __init__(self,
                 seq_len,
                 motif_path,
                 num_of_response=256,
                 window_size=200,
                 stride=1,
                 pad_motif=4,
                 **kwargs):
        super().__init__()

        self.seq_len = seq_len
        self.encoder = RawTISFM(
            num_of_response=num_of_response,
            motif_path=motif_path,
            window_size=window_size,
            stride=stride,
            pad_motif=pad_motif
        )

        self.output_dim = num_of_response

    def forward(self, x, **kwargs):
        # Assumes x is (B, 4, L)
        return self.encoder(x)

    def get_last_embedding_dimension(self):
        dummy = torch.randn(2, 4, self.seq_len).to(next(self.parameters()).device)
        out = self(dummy)
        return out.shape[-1]

    def _preprocess(self, x):
        if isinstance(x, str):
            x = [x]

        if isinstance(x, list):
            vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
            one_hots = []
            for seq in x:
                mat = torch.zeros(4, self.seq_len, dtype=torch.float32)
                for i, base in enumerate(seq.upper()[:self.seq_len]):
                    if base in vocab:
                        mat[vocab[base], i] = 1.0
                one_hots.append(mat)
            x = torch.stack(one_hots)  # (B, 4, L)

        if not isinstance(x, torch.Tensor) or x.dim() != 3 or x.size(1) != 4:
            raise ValueError("Input must be one-hot encoded or a list of DNA strings.")
        return x
