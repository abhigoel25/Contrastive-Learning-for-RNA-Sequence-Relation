import torch
from hydra.utils import instantiate
from src.model.simclr import get_simclr_model
# from src.model import psi_regression
from src.model import psi_regression, psi_regression_bothIntronExon
from src.model import MTSpliceBCE, MTSpliceWresnet_bothIntronExon



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
        print(f"\033[1;31m⚠️⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️  DEBUG MODE ENABLED in {filename}:{lineno} —{message} REMEMBER TO REVERT!\033[0m")
        _warned_debug = True
############# DEBUG Message ###############

def load_encoder(config, root_path, result_dir):
    
    # # Get the SimCLR model using its own configß
    simclr_model = get_simclr_model(config)

    if config.aux_models.warm_start:
        # simclr_model.load_state_dict(torch.load("checkpoints/introns_cl/NTv2/199/best-checkpoint.ckpt")["state_dict"], strict=False)
        # simclr_ckpt = "/mnt/home/at3836/Contrastive_Learning/files/results/exprmnt_2025_05_04__11_29_05/weights/checkpoints/introns_cl/ResNet1D/199/best-checkpoint.ckpt"
        
        simclr_ckpt = f"{root_path}/files/results/{result_dir}/weights/checkpoints/introns_cl/{config.embedder._name_}/199/best-checkpoint.ckpt"

        ckpt = torch.load(simclr_ckpt)
        state_dict = ckpt["state_dict"]

        # REMOVE "model." prefix from all keys
        cleaned_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        missing, unexpected = simclr_model.load_state_dict(cleaned_state_dict, strict=False)
        # print()
    return simclr_model.encoder


def initialize_encoders_and_model(config, root_path):
    mode = config.aux_models.mode
    result_dirs = {
        "5p": config.aux_models["weights_fiveprime"],
        "3p": config.aux_models["weights_threeprime"],
        "exon": config.aux_models["weights_exon"],
        "mtsplice_weights": config.aux_models["mtsplice_weights"]
    }

    if mode == "3p":
        encoder = load_encoder(config, root_path, result_dirs["3p"])
        return psi_regression.PSIRegressionModel(encoder, config)

    elif mode == "5p":
        encoder = load_encoder(config, root_path, result_dirs["5p"])
        return psi_regression.PSIRegressionModel(encoder, config)

    elif mode == "intronexon" or mode == "intronOnly":      
        encoder_5p = load_encoder(config, root_path, result_dirs["5p"])
        encoder_3p = load_encoder(config, root_path, result_dirs["3p"])
        
        if mode == "intronOnly":
            encoder_exon = get_simclr_model(config).encoder  # randomly initialized
        else:
            encoder_exon = load_encoder(config, root_path, result_dirs["exon"])

        if config.aux_models.mtsplice_BCE:
            return MTSpliceWresnet_bothIntronExon.PSIRegressionModel(
                encoder_5p, encoder_3p, encoder_exon, config)

        return psi_regression_bothIntronExon.PSIRegressionModel(
        encoder_5p, encoder_3p, encoder_exon, config)
    
    elif mode == "mtsplice":
        encoder = load_encoder(config, root_path, result_dirs["mtsplice_weights"])
        if config.aux_models.mtsplice_BCE:
            # If using BCE loss, return a mtsplice like model
            return MTSpliceBCE.MTSpliceBCE(encoder, config)
        else:
            # If using regression, return the standard model
            return psi_regression.PSIRegressionModel(encoder, config)
    else:
        raise ValueError(f"❌ Unsupported aux_models.mode: {mode}")

    
    

    # reset_debug_warning()
    # debug_warning("modify the logic to use resnet with mtsplice finetune")
        
    # return MTSpliceWresnet_bothIntronExon.PSIRegressionModel(
    #     encoder_5p, encoder_3p, encoder_exon, config
    # )