"""
This files is used to submit files in the slurm
"""
import os
import time
import random

trimester = time.strftime("_%Y_%m_%d__%H_%M_%S")
def create_job_dir(dir="", fold_name = ""):
    if dir:
        job_path = os.path.join(dir, fold_name)
        os.mkdir(job_path)
    else:
        job_path = os.path.join(os.getcwd(), fold_name)
        if not os.path.exists(job_path):
            os.mkdir(job_path)
    return job_path


# --- program bash script (1 run; RUN_IDX provided by caller) ---
def create_prg_file(prg_file_path):
    # NOTE: RUN_IDX can be exported by the caller; we default to 1 if unset
    header = f"""#!/bin/bash
    set -e
    cd $HOME
    source ~/.bashrc
    conda activate cl_splicing_regulation3
    : "${{RUN_IDX:=1}}"
    : "${{NEW_WANDB_PROJECT:=0}}"    # 1 => one new W&B project for ALL runs in this job
    PROJECT_NAME="{server_name}{slurm_file_name}{trimester}"   # no _run_ suffix
    EXTRA_WANDB_PROJECT=""
    if [ "$NEW_WANDB_PROJECT" = "1" ]; then
      EXTRA_WANDB_PROJECT=++logger.project="$PROJECT_NAME"
    fi
    WORKDIR={data_dir}
    cd $WORKDIR
    python -m scripts.psi_regression_training \\
            task={task} \\
            task.val_check_interval={val_check_interval}\\
            task.global_batch_size={global_batch_size}\\
            trainer.max_epochs={max_epochs}\\
            tokenizer={tokenizer} \\
            tokenizer.seq_len={tokenizer_seq_len} \\
            embedder={embedder} \\
            loss={loss_name} \\
            embedder.maxpooling={maxpooling} \\
            optimizer={optimizer} \\
            optimizer.lr={learning_rate} \\
            aux_models.freeze_encoder={freeze_encoder} \\
            aux_models.weights_fiveprime={weights_5p} \\
            aux_models.weights_threeprime={weight_3p} \\
            aux_models.weights_exon={weight_exon} \\
            aux_models.train_mode={train_mode} \\
            aux_models.eval_weights={eval_weights} \\
            aux_models.warm_start={warm_start} \\
            aux_models.mtsplice_weights={mtsplice_weights} \\
            aux_models.mode={mode} \\
            aux_models.mtsplice_BCE={mtsplice_BCE} \\
            dataset.test_files.intronexon={test_file} \\
            ++wandb.dir="'{wandb_dir}'"\\
            $EXTRA_WANDB_PROJECT \\
            ++logger.name="'{server_name}{slurm_file_name}{trimester}_run_$RUN_IDX'"\\
            ++callbacks.model_checkpoint.dirpath="'{checkpoint_dir}/run_$RUN_IDX'"\\
            ++hydra.run.dir="{hydra_dir}/run_$RUN_IDX"\\
            ++logger.notes="{wandb_logger_NOTES}"
    """
    # ++logger.project="'{server_name}{slurm_file_name}{trimester}_run_$RUN_IDX'" \\
    with open(prg_file_path, "w") as f:
        f.write(header)
    return prg_file_path


def copy_weights(desired_weight_path, to_be_saved_path ):
    for file_name in os.listdir(desired_weight_path):
        path = os.path.join(desired_weight_path, file_name)
        os.system(f"cp {path} {to_be_saved_path}")


def create_slurm_file(prg_file_path, job_name, slurm_file_path):
    show_name = '_'.join(job_name.split('_')[1:])
    show_name = f"{slurm_file_name}_{show_name}"
    header = f"#!/bin/bash\nbash {prg_file_path}"
    with open (slurm_file_path, "w") as f:
        f.write(header)
    return slurm_file_path


def get_file_name(kind, l0=0, l1=0, l2=0, l3=0, ext=True):
    file_name = f"{kind}_{trimester}"
    if ext:
        file_name = f"{file_name}.sh"
    return file_name

""" Parameters: **CHANGE (AT)** """
running_platform = 'EMPRAI'  # 'NYGC' or 'EMPRAI'
slurm_file_name = 'Psi_ASCOT_CL'  
gpu_num = 1
hour = 1
memory = 100 # GB
nthred = 8 # number of CPU
task = "psi_regression_task" 
val_check_interval = 1.0
global_batch_size = 8192
embedder = "mtsplice"
tokenizer = "onehot_tokenizer"
loss_name =  "MTSpliceBCELoss" # "multitissue_MSE" "MTSpliceBCELoss"
max_epochs = 20
maxpooling = True
optimizer = "sgd"
tokenizer_seq_len = 400
learning_rate =  1e-3
freeze_encoder = False
warm_start = True
mtsplice_weights = "exprmnt_2025_09_23__00_38_41"
# mtsplice_weights = "exprmnt_2025_07_30__13_10_26" #2 aug intronexon
# mtsplice_weights = "exprmnt_2025_08_16__22_30_50" #2 aug intron
# mtsplice_weights = "exprmnt_2025_08_16__20_42_52" #10 aug intronexon
# mtsplice_weights = "exprmnt_2025_08_23__20_30_33" #10 aug intron
# 2 aug
# weights_5p = "exprmnt_2025_06_01__21_15_08"
# weight_3p = "exprmnt_2025_06_01__21_16_19"
# weight_exon = "exprmnt_2025_06_08__21_34_21"
# 10 aug
weights_5p = "exprmnt_2025_08_23__21_20_33"
weight_3p = "exprmnt_2025_07_08__20_39_38"
weight_exon = "exprmnt_2025_08_23__21_20_33"
mode =  "mtsplice" # or "3p", "5p", "intronOnly", "intronexon", "mtsplice"
mtsplice_BCE = 1
train_mode = "train" # "train" for training, "eval" for evaluation only
eval_weights = "exprmnt_2025_08_26__17_46_02"
run_num = 1
TEST_FILE = "psi_variable_Retina___Eye_psi_MERGED.pkl"
readme_comment = "CL is trained on ASCOT dataset and weighted CL of 28k alternating exons"
wandb_logger_NOTES="CL is trained on ASCOT dataset and weighted CL of overlapping alternating exons"
new_project_wandb = 0 # if you want to create a new project for serial run
""" Parameters: **CHANGE (AT)** """ 

if running_platform == 'NYGC':
    server_name = 'NYGC'
    server_path = '/gpfs/commons/home/atalukder/'
elif running_platform == 'EMPRAI':
    server_name = 'EMPRAI'
    server_path = '/mnt/home/at3836/'

main_data_dir = server_path+"Contrastive_Learning/files/results"
job_path = server_path+"Contrastive_Learning/files/cluster_job_submission_files"
code_dir = server_path+"Contrastive_Learning/code/ML_model"

data_dir_0   = create_job_dir(dir= main_data_dir, fold_name= "exprmnt"+trimester)
data_dir     = create_job_dir(dir= data_dir_0, fold_name= "files")
weight_dir   = create_job_dir(dir= data_dir_0, fold_name="weights")
output_dir   = create_job_dir(dir= data_dir, fold_name="output_files")
hydra_dir    = create_job_dir(dir= data_dir, fold_name="hydra")
checkpoint_dir = create_job_dir(dir= weight_dir, fold_name="checkpoints")
wandb_dir    = create_job_dir(dir= data_dir, fold_name="wandb")

test_file = server_path + "Contrastive_Learning/data/final_data/ASCOT_finetuning//" + TEST_FILE
name = slurm_file_name

def create_readme():
    p = os.path.join(data_dir, "readme")
    with open(p, "a") as f:
        f.write(readme_comment)

create_readme()

def gen_combination(i):
    kind = name
    hash_obj = random.getrandbits(25)

    prg_file_path   = os.path.join(job_path, get_file_name(kind= f"prg_{kind}_{hash_obj}"))
    slurm_file_path = os.path.join(job_path, get_file_name(kind= f"slurm_{kind}_{hash_obj}"))

    create_prg_file(prg_file_path=prg_file_path)

    job_name = get_file_name(kind=f"{kind}_{hash_obj}", ext=False)

    # keep a code snapshot (optional)
    os.system(f"cp -r {code_dir}/scripts {data_dir}")
    os.system(f"cp -r {code_dir}/configs {data_dir}")
    os.system(f"cp -r {code_dir}/src {data_dir}")

    os.system(f"chmod u+x {prg_file_path}")

    # ★ Run sequentially in the same terminal session with per-run RUN_IDX
    #    and unique stdout per run:
    os.system(f"RUN_IDX={i} NEW_WANDB_PROJECT={new_project_wandb} bash {prg_file_path} 2>&1 | tee {output_dir}/out_{job_name}_r{i}.txt")

import time

def _fmt_secs(s: float) -> str:
    # hh:mm:ss for readability (handles long runs too)
    h, r = divmod(int(s), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    durations = []
    t_all = time.perf_counter()
    for i in range(1, run_num + 1):
        wall_start = time.strftime('%F %T')
        t0 = time.perf_counter()

        print(f"===== Starting RUN {i}/{run_num} at {wall_start} =====")
        gen_combination(i)

        dt = time.perf_counter() - t0
        wall_end = time.strftime('%F %T')
        print(f"===== Finished RUN {i}/{run_num} at {wall_end} (elapsed { _fmt_secs(dt) }) =====")
        durations.append(dt)

    total = time.perf_counter() - t_all
    if durations:
        mean_dt = sum(durations) / len(durations)
        med_dt = sorted(durations)[len(durations)//2]
        print(f"Total time: { _fmt_secs(total) } | mean/run: { _fmt_secs(mean_dt) } | median/run: { _fmt_secs(med_dt) }")

if __name__ == "__main__":
    main()
