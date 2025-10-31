import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Custom Dataset for the pairs of sequences using a Hugging Face tokenizer
class NucleotideSequencePairDataset(Dataset):
    """
    A custom dataset for handling pairs of nucleotide sequences. Each pair of sequences 
    is tokenized using a Hugging Face tokenizer, which is expected to convert each 
    sequence into token IDs in tensor format.
    """
    def __init__(self, sequences_1, sequences_2, tokenizer):
        """
        Initializes the dataset with two lists of nucleotide sequences and a tokenizer.

        Parameters:
        -----------
        sequences_1 : list of str
            A list containing the first set of nucleotide sequences.
        sequences_2 : list of str
            A list containing the second set of nucleotide sequences.
        tokenizer : Hugging Face tokenizer
            The tokenizer used to convert the sequences into token IDs.
        """
        self.sequences_1 = sequences_1
        self.sequences_2 = sequences_2
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.sequences_1)
    
    def __getitem__(self, idx):
        # Get both sequences for the pair
        seq1 = self.sequences_1[idx]
        seq2 = self.sequences_2[idx]
        
        # Tokenize the sequences
        view0 = self.tokenizer(seq1, return_tensors="pt")['input_ids'].squeeze(0)
        view1 = self.tokenizer(seq2, return_tensors="pt")['input_ids'].squeeze(0)
    
        
        return view0, view1
