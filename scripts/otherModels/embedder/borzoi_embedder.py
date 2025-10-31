import numpy as np
from baskerville import dna
from baskerville import seqnn
from .base_embedder import BaseEmbedder
import json


class BorzoiEmbedder(BaseEmbedder):
    def __init__(self, model_path, params_file = '/home/atalukder/Contrastive_Learning/models/borzoi/examples/params.json',seq_len=100):
        with open(params_file) as params_open :
    
            params = json.load(params_open)
            
            params_model = params['model']
            params_train = params['train']
        
        self.seq_len = seq_len
        params_model['seq_length'] = self.seq_len
        self.model = seqnn.SeqNN(params_model)
        self.model.restore(model_path, 0)

    def tokenize(self, seq: str):
        return dna.dna_1hot(seq.upper(), self.seq_len)[np.newaxis, :, :]

    def embed(self, seqs: list[str]):
        onehots = np.stack([self.tokenize(seq)[0] for seq in seqs])
        print("[DEBUG] onehots shape:", onehots.shape)
        preds = self.model.predict(onehots)
        print("[DEBUG] preds shape:", preds.shape)
        return preds
