import wandb
import numpy as np
import matplotlib.pyplot as plt

wandb.login()
api = wandb.Api()

folders = {
    "Mtsplice intron2aug": "NYGCPsi_mtspliceIntron2aug_2025_08_27__18_25_16",
    "Mtsplice intron10aug": "NYGCPsi_mtspliceIntron10aug_2025_08_27__18_26_08",
    "Mtsplice 2aug": "NYGCPsi_mtsplice2aug_2025_08_26__17_46_02",
    "Mtsplice 10aug": "NYGCPsi_mtsplice10aug_2025_08_26__17_50_05",
    "Resnet intronexon2Aug": "NYGCPsi_ResnetIntronexon2Aug_2025_08_27__17_11_29",
    "Resnet intronexon10Aug": "NYGCPsi_ResnetIntronexon10Aug_2025_08_27__17_20_16",
    "Resnet intron2aug": "NYGCPsi_resnetIntron2aug_2025_08_26__17_32_40",
    "Resnet intron10aug": "NYGCPsi_resnetIntron10aug_2025_08_26__17_30_10",
    "EMPRAI mtsplice ASCOT": "EMPRAIPsi_mtsplice_ASCOT_CL_2025_09_23__16_11_33",
    "Mtsplice Original": "NYGCPsi_mtspliceOriginal_2025_08_26__17_44_28",
}

nrows = len(folders) // 2 + len(folders) % 2
fig, axes = plt.subplots(nrows, 2, figsize=(12, 3*nrows))
axes = axes.flatten()

for idx, (label, folder_name) in enumerate(folders.items()):
    project = f"at3836-columbia-university/{folder_name}"
    runs = api.runs(project)

    x_vals, y_vals = [], []

    for run in runs:

        # scan_history streams ALL rows (not capped at 1000)
        min_val_loss = None
        i = 0
        for row in run.scan_history(keys=["val_loss_epoch"]):
            i += 1
            val = row.get("val_loss_epoch")
            if val is not None:
                if min_val_loss is None or val < min_val_loss:
                    min_val_loss = val
        
        if min_val_loss is not None:
            x_vals.append(min_val_loss)   # min val loss
            y_vals.append(run.summary.get("test_mean_spearman_delta"))   # delta at that min loss



        # hist = run.history(keys=["val_loss_epoch", "test_mean_spearman_delta"])
        # if "val_loss_epoch" not in hist or len(hist) == 0:
        #     continue

        # # find epoch with minimum validation loss
        # min_idx = hist["val_loss_epoch"].idxmin()
        # min_val_loss = hist.loc[min_idx, "val_loss_epoch"]

        # # get delta at that epoch (if logged)
        # delta = hist.loc[min_idx, "test_mean_spearman_delta"] if "test_mean_spearman_delta" in hist else None
        # if delta is not None:
        #     x_vals.append(min_val_loss)
        #     y_vals.append(delta)

    ax = axes[idx]
    if len(x_vals) > 0:
        # sort runs by val_loss
        order = np.argsort(x_vals)
        x_sorted = np.array(x_vals)[order]
        y_sorted = np.array(y_vals)[order]

        ax.plot(x_sorted, y_sorted, marker="o", linestyle="-", color="steelblue")
        ax.set_title(f"{label}\nN={len(x_sorted)}")
    else:
        ax.set_title(f"{label}\n(no valid runs)")

    ax.set_xlabel("Lowest val_loss_epoch (per run)")
    ax.set_ylabel("test_mean_spearman_delta")

# remove empty axes if odd number of projects
for j in range(idx+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()

plt.savefig("/gpfs/commons/home/atalukder/Contrastive_Learning/code/ML_model/scripts/figures/lossVSdeltamean_WandB_histograms.png")
