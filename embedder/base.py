import torch
import torch.nn as nn


class BaseEmbedder(nn.Module): #rename to BaseEmbedder
    """
    A class to handle the DNA embedding backbone.

    Args:
        model_name (str): Name of the model used for DNA embedding.
        rcps (bool): Whether to use reverse complement processing.
    """
    def __init__(self,
                 name_or_path,
                 bp_per_token,
                 rcps = False,
                 backbone=None,
                 _name_ = None,
                 ):
        super().__init__()
        self.name_or_path = name_or_path
        self.bp_per_token = bp_per_token
        self.backbone = backbone
        self.rcps = rcps


    def forward(self, input_ids, **kwargs):
        """Extract embeddings from the input IDs"""
        return self.backbone(input_ids, **kwargs)

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

        # Try to determine the input shape based on the first layer of the model
        for module in self.backbone.modules():
            if isinstance(module, nn.Conv2d):
                # Assume a common image input size if it's a Conv2d layer
                input_shape = (3, 224, 224)  # RGB image of size 224x224
                break
            elif isinstance(module, nn.Linear):
                # Assume a 1D input size for a fully connected layer
                input_shape = (module.in_features,)
                break
            elif isinstance(module, nn.Embedding):
                # Assume a single index for an Embedding layer
                input_shape = (64,)
                break
        else:
            raise ValueError("Unable to determine the input shape automatically.")

        # Generate a random input tensor and move it to GPU
        if isinstance(self.backbone, nn.Sequential):
            DEVICE = next(self.backbone.parameters()).device
        else:
            DEVICE = self.backbone.device
        random_input = torch.randint(low=0, high=2, size=(10, *input_shape)).to(DEVICE)
        
        # Pass the tensor through the model with no gradients
        with torch.no_grad():
            print(f"input shape: {random_input.shape}")
            if "nucleotide-transformer" in self.name_or_path:
                output = self(random_input)
                print(f"Initial output from the model: {output}")
            else:
                output = self(random_input)
                
            print(f"Output of the model of shape: {output.shape}")
            
            #Expted output of shape (batch_size, seq_len, embedding_dim)
            
        # Get the shape of the output tensor
        last_embedding_dimension = output.shape[-1]
        # Return the last dimension of the output tensor
        print(f"Found a last embedding dimension of {last_embedding_dimension}")
        return last_embedding_dimension
