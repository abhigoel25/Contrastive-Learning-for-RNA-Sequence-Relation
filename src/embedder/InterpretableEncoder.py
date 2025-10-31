import torch
import torch.nn as nn
import torch.nn.functional as F
from src.embedder.base import BaseEmbedder


class ReverseComplement1D(nn.Module):
    def __init__(self, complement=[3, 2, 1, 0]):
        super().__init__()
        self.complement = complement

    def forward(self, x):
        rev = torch.flip(x, dims=[-1])            # reverse along sequence
        revcom = rev[:, self.complement, :]       # reverse complement
        return revcom


class PWMConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, bias=False)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)


class TrainableScaling(nn.Module):
    def __init__(self, seq_len, channels):
        super().__init__()
        self.center = nn.Parameter(torch.zeros(1, channels))
        self.scale = nn.Parameter(torch.ones(1, channels))
        self.seq_len = seq_len

    def forward(self, x):
        C = self.center.repeat(self.seq_len, 1).T.unsqueeze(0)  # (1, C, L)
        S = self.scale.repeat(self.seq_len, 1).T.unsqueeze(0)  # (1, C, L)
        return S * x - C


class TrainablePooling(nn.Module):
    def __init__(self, seq_len, channels, initial_alpha=2.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1, channels), initial_alpha))
        self.seq_len = seq_len

    def forward(self, x):
        A = self.alpha.repeat(self.seq_len, 1).T.unsqueeze(0)  # (1, C, L)
        W = F.softmax(A * x, dim=-1)
        return torch.sum(W * x, dim=-1)  # (B, C)


class InterpretableEncoder1D(BaseEmbedder):
    def __init__(self,
                 seq_len=500,
                 in_channels=4,
                 motif_dim=256,
                 motif_width=12,
                 pooling_window=10,
                 **kwargs):
        super().__init__(name_or_path="InterpretableEncoder1D", bp_per_token=kwargs.get('bp_per_token', None))

        self.seq_len = seq_len  # NEW: for preprocessing
        self.dropout = nn.Dropout(p=0.2)

        self.rc = ReverseComplement1D()
        self.pwm_conv = PWMConv1D(in_channels, motif_dim, kernel_size=motif_width)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.pooling_window = pooling_window
        reduced_len = seq_len - motif_width + 1
        pooled_len = reduced_len - 1  # after MaxPool1d(kernel=2, stride=1)

        self.trainable_scaling = TrainableScaling(seq_len=pooled_len, channels=motif_dim)
        self.activation = nn.Sigmoid()
        self.trainable_pooling = TrainablePooling(seq_len=pooled_len, channels=motif_dim)

        self.interaction = nn.Linear(motif_dim, motif_dim, bias=False)
        nn.init.eye_(self.interaction.weight)

        self.batch_norm = nn.BatchNorm1d(motif_dim)
        self.output_dim = motif_dim

    
    def get_input_dtype_device(self):
        return self.pwm_conv.conv.weight.dtype, self.pwm_conv.conv.weight.device


    def forward(self, x, **kwargs):
        
        x = x.to(dtype=self.pwm_conv.conv.weight.dtype, device=self.pwm_conv.conv.weight.device)

        x_fwd = x
        x = self.pwm_conv(x_fwd) # need constraints
        x = self.maxpool(x)
        x = self.trainable_scaling(x)
        x = self.activation(x)
        x = self.dropout(x)               # Dropout on pre-pooled activations
        matches = self.trainable_pooling(x)
        weights = self.interaction(matches)
        weights = torch.sigmoid(weights)
        x = weights * matches
        # x = self.dropout(x)               # Dropout on reweighted summary
        x = self.batch_norm(x)
        return x


        """


    def forward(self, x, **kwargs):
        # dtype, device = self.get_input_dtype_device()
        # import time
        # start = time.time()
        # x = self._preprocess(x).to(dtype=dtype, device=device)
        # print(f"⏱️ preprocessing took {time.time() - start:.2f}s")
        x = x.to(dtype=self.pwm_conv.conv.weight.dtype, device=self.pwm_conv.conv.weight.device)

        x_fwd = x
        x_rc = self.rc(x_fwd)
        x_fwd_conv = self.pwm_conv(x_fwd)
        x_rc = self.rc(x_fwd) # check if tokenization and reverse is same
        x_fwd_conv = self.pwm_conv(x_fwd) # need constraints
        x_rc_conv = self.pwm_conv(x_rc)

        x_rc_conv = torch.flip(x_rc_conv, dims=[-1])  # reverse to align
        x = torch.maximum(x_fwd_conv, x_rc_conv)  # best match per strand

        x = self.maxpool(x)
        x = self.trainable_scaling(x)
        x = self.activation(x)
        x = self.trainable_pooling(x)

        x = self.interaction(x)
        x = self.batch_norm(x)
        return x

"""


        # x = x.to(dtype=self.pwm_conv.conv.weight.dtype, device=self.pwm_conv.conv.weight.device)

        # x_fwd = x
        # x = self.pwm_conv(x_fwd) # need constraints
        # x = self.maxpool(x)
        # x = self.trainable_scaling(x)
        # x = self.activation(x)
        # x = self.trainable_pooling(x)

        # x = self.interaction(x)
        # x = self.batch_norm(x)
        # return x

        


    def get_last_embedding_dimension(self):
        with torch.no_grad():
            # dummy = torch.randn(2, 4, self.seq_len).to(next(self.parameters()).device)
            dummy = torch.full((2, 4, self.seq_len), 10.0).to(next(self.parameters()).device)
            out = self(dummy)
        print(f"Interpretable encoder output dim: {out.shape[-1]}")
        return out.shape[-1]

    def _preprocess(self, x):
        """
        Accepts:
            - List[str] or str (DNA sequences)
            - torch.Tensor of shape (B, 4, L)
        Returns:
            - torch.Tensor of shape (B, 4, L)
        """
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
            raise ValueError("Input must be one-hot encoded or a string/list of DNA sequences.")

        return x





