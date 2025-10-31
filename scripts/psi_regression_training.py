import sys
import os
import subprocess
import torch
from pathlib import Path
from omegaconf import OmegaConf

# Add the parent directory (main) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def find_contrastive_root(start: Path = Path(__file__)) -> Path:
    for parent in start.resolve().parents:
        if parent.name == "Contrastive_Learning":
            return parent
    raise RuntimeError("Could not find 'Contrastive_Learning' directory.")

# Set env var *before* hydra loads config
# os.environ["CONTRASTIVE_ROOT"] = str(find_contrastive_root())
root_path = str(find_contrastive_root())
os.environ["CONTRASTIVE_ROOT"] = root_path


import hydra
from omegaconf import OmegaConf
import torch
# from src.model.psi_regression import PSIRegressionModel
from src.trainer.utils import create_trainer
from src.datasets.auxiliary_jobs import PSIRegressionDataModule
# from src.model.simclr import get_simclr_model
from src.utils.config import  print_config
from src.utils.encoder_init import initialize_encoders_and_model


import importlib

def load_model_class(config):
    module_name = f"src.model.{config.aux_models.model_script}"
    module = importlib.import_module(module_name)
    return getattr(module, "PSIRegressionModel")

# os.environ['WANDB_INIT_TIMEOUT'] = '600'
def get_optimal_num_workers():
    num_cpus = os.cpu_count()
    num_gpus = torch.cuda.device_count()
    return min(num_cpus // max(1, num_gpus), 16)




@hydra.main(version_base=None, config_path="../configs", config_name="psi_regression.yaml")
def main(config: OmegaConf):

    def get_free_gpu():
        result = subprocess.check_output(
            "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader", shell=True
        )
        memory_used = [int(x) for x in result.decode("utf-8").strip().split("\n")]
        return memory_used.index(min(memory_used))

    # Choose and set GPU
    free_gpu = get_free_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpu)
    print(f"Using GPU {free_gpu}: {torch.cuda.get_device_name(0)}")


    # Register Hydra resolvers
    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
    OmegaConf.register_new_resolver('min', lambda x, y: min([x, y]))
    OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
    OmegaConf.register_new_resolver('optimal_workers', lambda: get_optimal_num_workers())

    # result_dir = config.task.pretraining_weights

    
    # Print and process configuration
    print_config(config, resolve=True)

    # Initialize the IntronsDataModule  #####(AT)######
    data_module = PSIRegressionDataModule(config)
    data_module.prepare_data()
    data_module.setup()

    # EVAL-ONLY: load checkpoint and run test
    mode = str(getattr(config.aux_models, "train_mode", "train")).lower()
    ckpt_path = getattr(config.aux_models, "eval_weights", None)
    if mode == "eval":
        
        config.aux_models.warm_start = False
        model = initialize_encoders_and_model(config, root_path)
        print(f"[Eval] Loading checkpoint: {ckpt_path}")
        ckpt_path = f"{root_path}/files/results/{ckpt_path}/weights/checkpoints/{config.task._name_}/{config.embedder._name_}/201/best-checkpoint.ckpt"
        trainer = create_trainer(config)   
        trainer.test(model=model, ckpt_path=ckpt_path, datamodule=data_module)
        return


    model = initialize_encoders_and_model(config, root_path)


    # Traingi
    trainer = create_trainer(config)
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())
    trainer.test(model, datamodule=data_module)

    print('######END#######')
    

if __name__ == "__main__":
    main()



    