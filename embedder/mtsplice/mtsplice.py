import inspect
import os
import torch
import torch.nn as nn
from src.embedder.base import BaseEmbedder # Assuming this exists
from src.embedder.mtsplice.splinetransformer import SplineWeight1D  # Assuming this is the correct import path


############# DEBUG Message ###############
import inspect
import os
_warned_debug = False  # module-level flag
def reset_debug_warning():
    global _warned_debug
    _warned_debug = False
def debug_warning(message):
    global _warned_debug
    if not _warned_debug:
        frame = inspect.currentframe().f_back
        filename = os.path.basename(frame.f_code.co_filename)
        lineno = frame.f_lineno
        print(f"\033[1;31m⚠️⚠️ ⚠️ ⚠️ DEBUG MODE ENABLED in {filename}:{lineno} —{message} REMEMBER TO REVERT!\033[0m")
        _warned_debug = True
############# DEBUG Message ###############


        
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, padding='same'),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return x + self.block(x)


class MTSpliceBranch(nn.Module):
    def __init__(self, seq_len, in_channels=4, hidden_channels=64, num_blocks=9, spline_kwargs={}):
        super().__init__()
        layers = [
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1, padding='same'),
            nn.BatchNorm1d(hidden_channels)
        ]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(hidden_channels))
        self.resnet = nn.Sequential(*layers)

        # Replace placeholder with the real SplineWeight1D layer
        self.spline = SplineWeight1D(
            input_steps=seq_len,
            input_filters=hidden_channels,
            **spline_kwargs
        )

    def forward(self, x):
        # Input x: (B, Length, Channels), e.g., (B, 400, 4)
        # x = x.permute(0, 2, 1)  # (B, C, L) for Conv1d
        x  = x.float()
        x = self.resnet(x)      # (B, hidden_C, L)
        
        # Permute for Spline layer, apply it, and permute back
        x = x.permute(0, 2, 1)  # (B, L, hidden_C) for spline
        x = self.spline(x)
        x = x.permute(0, 2, 1)  # (B, hidden_C, L) for main encoder
        return x


class MTSpliceEncoder(BaseEmbedder):
    # (AT) seq_len=400 in the origianal paper but becaus we are trying with intron only so 200
    def __init__(self, seq_len=400, in_channels=4, hidden_dim=64, embed_dim=32, out_dim=56, dropout=0.5, spline_kwargs={}, **kwargs):
        super().__init__(name_or_path="MTSplice", bp_per_token=kwargs.get('bp_per_token', None))

        # Pass seq_len and spline_kwargs to each branch
        self.branch_l = MTSpliceBranch(seq_len, in_channels, hidden_dim, spline_kwargs=spline_kwargs)
        self.branch_r = MTSpliceBranch(seq_len, in_channels, hidden_dim, spline_kwargs=spline_kwargs)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.bn1 = nn.BatchNorm1d(hidden_dim) # *2 because we concatenate hidden_dim from both branches
        self.fc1 = nn.Linear(hidden_dim, embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_dim, out_dim)

    def forward(self, seql, seqr, **kwargs):
        x_l = self.branch_l(seql)  # Output shape: (B, hidden_dim, 400)
        x_r = self.branch_r(seqr)  # Output shape: (B, hidden_dim, 400)

        # To match Keras, concatenate along the sequence length dimension (dim=2)
        # Note: The input to your BatchNorm/Linear layers must be changed to `hidden_dim` instead of `hidden_dim * 2`
        x = torch.cat([x_l, x_r], dim=2)  # -> (B, hidden_dim, 800)

        # Now, apply global average pooling
        x = self.global_pool(x).squeeze(-1)  # -> (B, hidden_dim)

        # The rest of your head architecture
        # Ensure self.bn1 and self.fc1 are initialized with `hidden_dim`
        x = self.bn1(x)
        # x = self.fc1(x)
        # x = self.bn2(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        return x
        
    def get_regularization_loss(self):
        """Call this in your training_step to add the spline penalty."""
        return self.branch_l.spline.get_regularization_loss() + \
               self.branch_r.spline.get_regularization_loss()
    

    def get_last_embedding_dimension(self) -> int:
        """
        Computes the final embedding output dimension of the MTSpliceEncoder
        by passing dummy inputs through the full forward pass (seql + seqr).

        Returns:
            int: The last dimension of the encoder output.
        """
        DEVICE = next(self.parameters()).device
        seq_len = self.branch_l.seq_len if hasattr(self.branch_l, 'seq_len') else 400
        in_channels = self.branch_l.in_channels if hasattr(self.branch_l, 'in_channels') else 4

        # Create dummy input tensors: shape (batch_size=10, in_channels=4, seq_len=400)
        dummy_seql = torch.randn(10, in_channels, seq_len).to(DEVICE)
        dummy_seqr = torch.randn(10, in_channels, seq_len).to(DEVICE)

        with torch.no_grad():
            output = self(dummy_seql, dummy_seqr)

        last_dim = output.shape[-1]
        print(f"[MTSpliceEncoder] Last embedding dimension: {last_dim}")
        return last_dim
