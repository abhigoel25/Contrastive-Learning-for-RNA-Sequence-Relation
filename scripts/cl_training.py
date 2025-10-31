import sys
import os
from pathlib import Path
from omegaconf import OmegaConf
import os


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



# reset_debug_warning()
# debug_warning("all seed fixed")
        
# import os, random, numpy as np, torch
# seed = 42
# os.environ["PYTHONHASHSEED"] = str(seed)
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# from lightning import seed_everything
# seed_everything(seed, workers=True)


# Add the parent directory (main) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def find_contrastive_root(start: Path = Path(__file__)) -> Path:
    for parent in start.resolve().parents:
        if parent.name == "Contrastive_Learning":
            return parent
    raise RuntimeError("Could not find 'Contrastive_Learning' directory.")

# Set env var *before* hydra loads config
os.environ["CONTRASTIVE_ROOT"] = str(find_contrastive_root())
CONTRASTIVE_ROOT = find_contrastive_root()



import hydra
from omegaconf import OmegaConf
from src.utils.config import  print_config
import torch
from src.model.lit import create_lit_model
from src.trainer.utils import create_trainer
from src.datasets.lit import ContrastiveIntronsDataModule

# os.environ['WANDB_INIT_TIMEOUT'] = '600'
def get_optimal_num_workers():
    num_cpus = os.cpu_count()
    num_gpus = torch.cuda.device_count()
    return min(num_cpus // max(1, num_gpus), 8) #(AT) modified to run in empireAI
    # return 1


    # num_cpus = os.cpu_count()
    # suggested_max = 1  # Override based on warning
    # return min(num_cpus // max(1, torch.cuda.device_count()), suggested_max)


@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(config: OmegaConf):

    # Register Hydra resolvers
    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
    OmegaConf.register_new_resolver('min', lambda x, y: min([x, y]))
    OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
    OmegaConf.register_new_resolver('optimal_workers', lambda: get_optimal_num_workers())

    # Print and process configuration
    print_config(config, resolve=True)


    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())
    # print(torch.version.cuda)
    # print(torch.backends.cudnn.version())



    # Initialize the IntronsDataModule with dataset-specific configs
    data_module = ContrastiveIntronsDataModule(config)
    # data_module.prepare_data()
    data_module.setup()
    
    tokenizer = data_module.tokenizer
    
    lit_model = create_lit_model(config)
    
    trainer = create_trainer(config)
    
    trainer.fit(lit_model, data_module.train_dataloader(), data_module.val_dataloader())
if __name__ == "__main__":
    main()
