"""
This script launches training jobs in a persistent `screen` session.
No SLURM is used. Jobs are submitted using detached screen sessions.
"""
import os
import time
import random

trimester = time.strftime("_%Y_%m_%d__%H_%M_%S")

def create_job_dir(dir="", fold_name=""):
    if dir:
        job_path = os.path.join(dir, fold_name)
        os.mkdir(job_path)
    else:
        job_path = os.path.join(os.getcwd(), fold_name)
        if not os.path.exists(job_path):
            os.mkdir(job_path)
    return job_path

def create_prg_file(prg_file_path):
    header = f"""#!/bin/bash
set -e
OUTPUTDIR={output_dir}
exec > "$OUTPUTDIR/job_output.log" 2>&1
cd $HOME
eval "$(conda shell.bash hook)"
conda activate cl_splicing_regulation2
WORKDIR={data_dir}
cd $WORKDIR
python -m scripts.cl_training \\
        task=introns_cl \\
        task.val_check_interval=0.5 \\
        task.global_batch_size=4096 \\
        trainer.max_epochs=3 \\
        tokenizer="custom_tokenizer" \\
        embedder="resnet" \\
        embedder.maxpooling=True \\
        optimizer="sgd" \\
        ++wandb.dir="'{wandb_dir}'" \\
        ++logger.name="'{slurm_file_name}{trimester}'" \\
        ++callbacks.model_checkpoint.dirpath="'{checkpoint_dir}'" \\
        ++hydra.run.dir={hydra_dir} \\
        ++logger.notes="{wandb_logger_NOTES}"
"""
    with open(prg_file_path, "w") as f:
        f.write(header)
    return prg_file_path

def get_file_name(kind, ext=True):
    file_name = f"{kind}_{trimester}"
    if ext:
        file_name = f"{file_name}.sh"
    return file_name

def create_readme():
    name = os.path.join(data_dir, "readme")
    with open(name, "a") as readme:
        readme.write(readme_comment)

def gen_combination():
    create_readme()

    kind = slurm_file_name
    hash_obj = random.getrandbits(25)

    prg_file_path = os.path.join(job_path, get_file_name(kind=f"prg_{kind}_{hash_obj}"))
    create_prg_file(prg_file_path=prg_file_path)

    os.system(f"cp -r {code_dir}/scripts {data_dir}")
    os.system(f"cp -r {code_dir}/configs {data_dir}")
    os.system(f"cp -r {code_dir}/src {data_dir}")

    os.system(f"chmod u+x {prg_file_path}")
    os.system(f"screen -dmS {slurm_file_name}_{hash_obj} bash {prg_file_path}")

def main():
    gen_combination()

if __name__ == "__main__":
    """ Environment and parameters """
    # primary_dir = '/home/argha/'
    primary_dir = '/mnt/home/at3836/'
    main_data_dir = primary_dir+"Contrastive_Learning/files/results"
    job_path = primary_dir+"Contrastive_Learning/files/cluster_job_submission_files"
    code_dir = primary_dir+"Contrastive_Learning/code/ML_model"

    data_dir_0   = create_job_dir(dir=main_data_dir, fold_name="exprmnt" + trimester)
    data_dir     = create_job_dir(dir=data_dir_0, fold_name="files")
    weight_dir   = create_job_dir(dir=data_dir_0, fold_name="weights")
    output_dir   = create_job_dir(dir=data_dir, fold_name="output_files")
    hydra_dir    = create_job_dir(dir=data_dir, fold_name="hydra")
    checkpoint_dir = create_job_dir(dir=weight_dir, fold_name="checkpoints")
    wandb_dir    = create_job_dir(dir=data_dir, fold_name="wandb")

    slurm_file_name = 'SRVR'
    readme_comment = "Maxpool architecture, 50 epochs, 6000 batch size, loss ntxent"
    wandb_logger_NOTES = "server submission trial"

    main()
