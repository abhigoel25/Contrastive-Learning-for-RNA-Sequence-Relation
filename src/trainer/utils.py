from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import wandb
import os
import sys
from lightning.pytorch.loggers import WandbLogger


def create_trainer(config: OmegaConf):
        # Initialize the logger
        wandb.login(key=config.wandb.api_key)

        #Check on the checkpoint directory
        if not os.path.exists(config.callbacks.model_checkpoint.dirpath):
                print(f"Creating directory {config.callbacks.model_checkpoint.dirpath}")
                os.makedirs(config.callbacks.model_checkpoint.dirpath)
                
        # Instantiate callbacks
        callbacks = []
        for cb_name, cb_conf in config.callbacks.items():
                callbacks.append(instantiate(cb_conf))

        # # Explicitly initialize WandB Logger with increased timeout
        # wandb_logger = WandbLogger(
        #         settings=wandb.Settings(init_timeout=600)  # Set timeout to 10 minutes
        # )
        # # Instantiate the trainer with the WandB logger
        # trainer = instantiate(config.trainer, callbacks=callbacks, logger=wandb_logger)

        # wandb_logger = WandbLogger(
        #         name="my_run_name",                       # name of the run
        #         project="INTRONS_CL",                # W&B project name
        #         log_model=True,                           # optional: log checkpoints
        #         save_dir="./wandb/",                             # optional: save logs here
        #         settings=wandb.Settings(init_timeout=600) # timeout in seconds
        #         )
        # trainer = instantiate(config.trainer, callbacks=callbacks, logger=wandb_logger)

        # logger = instantiate(config.logger)    
        # trainer = instantiate(config.trainer, callbacks=callbacks, logger=logger)

        if config.logger._target_ == "lightning.pytorch.loggers.WandbLogger":
                wandb_logger = WandbLogger(
                        name=config.logger.name,
                        project=config.logger.project,
                        group=config.logger.group,
                        save_dir=config.logger.save_dir,
                        log_model=config.logger.log_model,
                        config=OmegaConf.to_container(config, resolve=True),  # ✅ this logs the config
                        settings=wandb.Settings(init_timeout=600)
                )
                # wandb_run = wandb_logger.experiment
                # wandb_run.config.update(OmegaConf.to_container(config, resolve=True), allow_val_change=True)

                logger = wandb_logger
        else:
                logger = instantiate(config.logger)
        trainer = instantiate(config.trainer, callbacks=callbacks, logger=logger)
           
        return trainer

        """
        # (AT)
        logger = instantiate(config.logger)      
        # Instantiate the trainer
        trainer = instantiate(config.trainer, callbacks=callbacks, logger=logger)
        return trainer
        """
        