"LongSafari/hyenadna-small-32k-seqlen"



from transformers import AutoModelForMaskedLM
from src.embedder.base import BaseEncoder

class HyenaEmbedder(BaseEncoder):
    
    def __init__(self, **kwargs):
        raise NotImplementedError("HyenaEmbedder is not implemented yet.")