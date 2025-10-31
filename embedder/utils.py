from src.embedder.ntv2 import NTv2Embedder
from src.embedder.resnet import ResNet1D
from src.embedder.resnet101 import ResNet1D101
# from src.embedder.interpretable_encoder import InterpretableEncoder1D  
# from src.embedder.tisfm_encoder import TISFMEncoder
from src.embedder.InterpretableEncoder import InterpretableEncoder1D  
from src.embedder.tisfm import TISFMEncoder
from src.embedder.mtsplice.mtsplice import MTSpliceEncoder # (AT)


EMBEDDERS = {
    'NTv2': NTv2Embedder,
    'ResNet1D': ResNet1D,
    'ResNet1D101': ResNet1D101,
    'InterpretableEncoder1D': InterpretableEncoder1D,
    'TISFM': TISFMEncoder,
    'MTSplice': MTSpliceEncoder # (AT)
}



def get_embedder(config):
    """
    Function to get the backbone model for the task.

    Args:
        config (OmegaConf): The config object.

    Returns:
        nn.Module: The backbone model.
    """
    # Get the backbone model
    embedder = EMBEDDERS[config.embedder._name_](**config.embedder)
    return embedder