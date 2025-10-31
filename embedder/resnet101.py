### modification from resnet_old1 is previously we inject maxpooling

# Bottleneck block adapted for 1D convolutions
import torch
import torch.nn as nn
from src.embedder.base import BaseEmbedder

class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # Adjust dimensions if needed
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)  # Reduce dimensions
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # Main convolution
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)  # Restore dimensions
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # Adjust identity

        out += identity  # Residual connection
        out = self.relu(out)

        return out

# ResNet-50 architecture adapted for 1D convolutions
class ResNet1D101(BaseEmbedder):
    def __init__(self,
                 layers=[3, 4, 23, 3],
                 vocab_size=11,
                 embedding_dim=768,
                 **kwargs):
        super().__init__(name_or_path="ResNet1D", bp_per_token=kwargs.get('bp_per_token', None))
        
        # self.in_channels = embedding_dim
        self.layers = layers
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.in_channels = 64
        self.use_maxpooling = kwargs.get("maxpooling", False)
        self.backbone = self.initialize_resnet()
        
    
    def initialize_resnet(self):
        # Embedding layer
        embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        conv1 = nn.Conv1d(
            in_channels=self.embedding_dim,
            out_channels=self.in_channels,     # usually 64
            kernel_size=7,
            stride=2,
            padding=3,                          # to keep "same" size when possible
            bias=False
        )
        bn1 = nn.BatchNorm1d(self.in_channels)
        relu = nn.ReLU(inplace=True)
            
        layer1 = self._make_layer(Bottleneck1D, 64, self.layers[0])
        layer2 = self._make_layer(Bottleneck1D, 128, self.layers[1], stride=1)
        layer3 = self._make_layer(Bottleneck1D, 256, self.layers[2], stride=1)
        layer4 = self._make_layer(Bottleneck1D, 512, self.layers[3], stride=1)


        # Combine all layers into the backbone
        backbone = nn.Sequential(
            embedding,
            conv1,
            bn1,
            relu,
            *([nn.MaxPool1d(kernel_size=3, stride=2, padding=1)] if self.use_maxpooling else []),  # after conv1

            layer1,
            *([nn.MaxPool1d(kernel_size=3, stride=2, padding=1)] if self.use_maxpooling else []),  # after layer1

            layer2,
            *([nn.MaxPool1d(kernel_size=3, stride=2, padding=1)] if self.use_maxpooling else []),  # after layer2

            layer3,
            *([nn.MaxPool1d(kernel_size=2, stride=2, padding=0)] if self.use_maxpooling else []),  # after layer3

            layer4,
            *([nn.MaxPool1d(kernel_size=2, stride=1, padding=0)] if self.use_maxpooling else []),  # after layer4 (optional)
        )

        return backbone
        

    def _make_layer(self, Bottleneck, out_channels, blocks, stride=1):
        downsample = None

        if stride != 1 or self.in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * Bottleneck.expansion)
            )

        layers = []
        layers.append(Bottleneck(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * Bottleneck.expansion

        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, input_ids, **kwargs):
        # Expect input_ids to be of shape (batch_size, sequence_length)
        
        x = self.backbone[0](input_ids)  # Embedding layer
        # Reshape to (batch_size, embedding_dim, sequence_length)
        x = x.permute(0, 2, 1)
        # Pass the reshaped input through the remaining layers of the backbone
        x = self.backbone[1:](x)
        x = x.mean(dim=2)
        # x=x.permute(0, 2, 1)
        return x
    
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
        # Assume a single index for an Embedding layer
        input_shape = (64,)

        DEVICE = next(self.backbone.parameters()).device
        random_input = torch.randint(low=0, high=2, size=(10, *input_shape)).to(DEVICE)
        
        # Pass the tensor through the model with no gradients
        with torch.no_grad():
            # print(f"input shape: {random_input.shape}")
            output = self(random_input)
            # print(f"Output of the model of shape: {output.shape}")
            
            #Expted output of shape (batch_size, seq_len, embedding_dim)
            
        # Get the shape of the output tensor
        last_embedding_dimension = output.shape[-1]
        # Return the last dimension of the output tensor
        print(f"Found a last embedding dimension of {last_embedding_dimension}")
        return last_embedding_dimension
