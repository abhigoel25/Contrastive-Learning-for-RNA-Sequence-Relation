import torch
from typing import List


class FastOneHotPreprocessor:
    def __init__(self, seq_len: int = 200, vocab: str = "ACGT", padding: str = "right"):
        self.vocab = vocab
        self.vocab_map = {ch: i for i, ch in enumerate(vocab)}
        self.seq_len = seq_len
        self.padding = padding  # "right" or "left"
        self.pad_id = len(vocab)

    def encode_batch(self, sequences: List[str]) -> torch.Tensor:
        # Create a lookup table: ASCII → vocab index
        lut = torch.full((128,), self.pad_id, dtype=torch.long)
        for ch, idx in self.vocab_map.items():
            lut[ord(ch)] = idx
            lut[ord(ch.lower())] = idx  # handle lowercase too

        # Convert list of strings to a flat string of ASCII codes
        batch_size = len(sequences)
        tensor = torch.full((batch_size, self.seq_len), self.pad_id, dtype=torch.long)

        for i, seq in enumerate(sequences):
            seq = seq[:self.seq_len]
            encoded = lut[torch.tensor([ord(c) for c in seq], dtype=torch.long)]
            L = len(encoded)
            if self.padding == "right":
                tensor[i, :L] = encoded
            else:
                tensor[i, -L:] = encoded

        return tensor

    def __call__(self, sequences: List[str]) -> torch.Tensor:
        indices = self.encode_batch(sequences)  # shape: (B, L)
        one_hot = torch.nn.functional.one_hot(indices, num_classes=self.pad_id + 1).movedim(-1, 1)
        return one_hot[:, :-1, :]  # drop padding/UNK channel




# import torch
# from typing import List


# class FastOneHotPreprocessor:
#     def __init__(self, seq_len: int = 200, vocab: str = "ACGT", padding: str = "right"):
#         self.vocab = {ch: i for i, ch in enumerate(vocab)}
#         self.seq_len = seq_len
#         self.padding = padding  # "right" or "left"

#     def __call__(self, sequences: List[str]) -> torch.Tensor:
#         pad_id = len(self.vocab)  # treat pad as [UNK] index
#         indices = torch.full((len(sequences), self.seq_len), fill_value=pad_id, dtype=torch.long)

#         for i, seq in enumerate(sequences):
#             seq = seq.upper()[:self.seq_len]
#             L = len(seq)
#             for j, base in enumerate(seq):
#                 idx = j if self.padding == "right" else self.seq_len - L + j
#                 indices[i, idx] = self.vocab.get(base, pad_id)

#         one_hot = torch.nn.functional.one_hot(indices, num_classes=pad_id + 1).movedim(-1, 1)
#         return one_hot[:, :-1, :]  # drop the [PAD]/UNK channel
