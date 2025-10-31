import wandb
import numpy as np

wandb.login()
api = wandb.Api()

# folder_name = "NYGCPsi_mtspliceIntron2aug_2025_08_27__18_25_16" # Mtsplice intron2aug
# folder_name = "NYGCPsi_mtspliceIntron10aug_2025_08_27__18_26_08" # Mtsplice intron10aug
# folder_name = "NYGCPsi_mtsplice2aug_2025_08_26__17_46_02" # 
# folder_name = "NYGCPsi_mtsplice10aug_2025_08_26__17_50_05"
# folder_name = "NYGCPsi_ResnetIntronexon2Aug_2025_08_27__17_11_29"
# folder_name = "NYGCPsi_ResnetIntronexon10Aug_2025_08_27__17_20_16"
# folder_name = "NYGCPsi_resnetIntron2aug_2025_08_26__17_32_40"
# folder_name = "NYGCPsi_resnetIntron10aug_2025_08_26__17_30_10"
# folder_name = "EMPRAIPsi_mtsplice_ASCOT_CL_2025_09_23__16_11_33"
folder_name = "NYGCPsi_mtspliceOriginal_2025_08_26__17_44_28"

project = f"at3836-columbia-university/{folder_name}"
runs = api.runs(project)

values = [run.summary.get("test_mean_spearman_delta") for run in runs 
          if run.summary.get("test_mean_spearman_delta") is not None]
values = np.array(values)


print("======================")
print(f"Project: {project}")
print("Number of runs:", len(values))
print("Mean:", values.mean())
print("Std (population, ddof=0):", np.std(values))
print("======================")
# print("Std (sample, ddof=1):", np.std(values, ddof=1))
