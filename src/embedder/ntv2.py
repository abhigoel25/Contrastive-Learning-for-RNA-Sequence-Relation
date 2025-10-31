import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM
from src.embedder.base import BaseEmbedder

class NTv2Embedder(BaseEmbedder):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = self.initialize_ntv2()
    
    def initialize_ntv2(self):
        backbone = AutoModelForMaskedLM.from_pretrained(self.name_or_path, trust_remote_code=True).esm
        backbone.contact_head = None
        if backbone.config.position_embedding_type == "rotary":
            backbone.embeddings.position_embeddings = None
        return backbone
    
    def forward(self, input_ids, **kwargs):
        """Extract embeddings from the input IDs"""
        return self.backbone(input_ids, **kwargs)[0]
    
    def get_last_embedding_dimension(self) -> int:
        """
        Function to get the last embedding dimension of a PyTorch model by passing
        a random tensor through the model and inspecting the output shape.
        This is done with gradients disabled and always on GPU.

        Args:
            model (nn.Module): The PyTorch model instance.

        Returns:
            int: The last embedding dimension (i.e., the last dimension of the output tensor).
        """
        DEVICE = self.backbone.device
        input_shape = (64,)
        random_input = torch.randint(low=0, high=2, size=(10, *input_shape)).to(DEVICE)
        
        # Pass the tensor through the model with no gradients
        with torch.no_grad():
            print(f"input shape: {random_input.shape}")
            output = self(random_input)
            print(f"Output of the model of shape: {output.shape}")
            
            #Expted output of shape (batch_size, seq_len, embedding_dim)
            
        # Get the shape of the output tensor
        last_embedding_dimension = output.shape[-1]
        # Return the last dimension of the output tensor
        print(f"Found a last embedding dimension of {last_embedding_dimension}")
        return last_embedding_dimension
