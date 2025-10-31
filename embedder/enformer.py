from enformer_pytorch import from_pretrained
from src.embedder.base import BaseEncoder


class EnformerEmbedder(BaseEncoder):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = from_pretrained(self.model_name, use_tf_gamma = False)