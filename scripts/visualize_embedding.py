import sys
import os
import torch
import torch.nn.functional as F
import random

# Add the parent directory (main) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hydra
from omegaconf import OmegaConf
from src.utils.config import  print_config
import torch
from src.model.lit import create_lit_model
from src.trainer.utils import create_trainer
from src.datasets.lit import ContrastiveIntronsDataModule

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import time


trimester = time.strftime("_%Y_%m_%d__%H_%M_%S")

# os.environ['WANDB_INIT_TIMEOUT'] = '600'
def get_optimal_num_workers():
    num_cpus = os.cpu_count()
    num_gpus = torch.cuda.device_count()
    return min(num_cpus // max(1, num_gpus), 16)

def get_2view_embedding(config, device, view0, view1):
    lit_model = create_lit_model(config)
    simclr_ckpt = "/gpfs/commons/home/atalukder/Contrastive_Learning/files/results/exprmnt_2025_04_05__22_51_31/weights/checkpoints/introns_cl/ResNet1D/199/best-checkpoint.ckpt"
    ckpt = torch.load(simclr_ckpt, map_location=device)
    state_dict = ckpt["state_dict"]
    cleaned_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    lit_model.load_state_dict(cleaned_state_dict, strict=False)
    lit_model.to(device)
    lit_model.eval()

   # Compute embeddings
    with torch.no_grad():
        z0 = lit_model(view0.to(device))
        z1 = lit_model(view1.to(device))

    # Combine and compute t-SNE
    embeddings = torch.cat([z0, z1], dim=0).cpu().numpy()
    tsne = TSNE(n_components=2, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings)

    z0_2d = emb_2d[:z0.shape[0]]
    z1_2d = emb_2d[z0.shape[0]:]

    # Plot with lines between pairs
    plt.figure(figsize=(8, 8))
    for i in range(z0.shape[0]):
        plt.plot(
            [z0_2d[i, 0], z1_2d[i, 0]],
            [z0_2d[i, 1], z1_2d[i, 1]],
            color="gray",
            linewidth=0.5,
            alpha=0.5
        )

    plt.scatter(z0_2d[:, 0], z0_2d[:, 1], color="blue", s=20, label="Anchor (z0)")
    plt.scatter(z1_2d[:, 0], z1_2d[:, 1], color="orange", s=20, label="Positive (z1)")
    plt.legend()
    plt.title("t-SNE: Anchor–Positive Pairs")
    plt.axis("off")
    plt.tight_layout()
    
    plt.savefig(f'/gpfs/commons/home/atalukder/Contrastive_Learning/code/ML_model/figures/tsne{trimester}.png')
    

def all_pos_of_anchor(config, device, view0, train_loader, tokenizer):
    # Load the pretrained model
    lit_model = create_lit_model(config)
    simclr_ckpt = "/gpfs/commons/home/atalukder/Contrastive_Learning/files/results/exprmnt_2025_04_05__22_51_31/weights/checkpoints/introns_cl/ResNet1D/199/best-checkpoint.ckpt"
    ckpt = torch.load(simclr_ckpt, map_location=device)
    state_dict = ckpt["state_dict"]
    cleaned_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    lit_model.load_state_dict(cleaned_state_dict, strict=False)
    lit_model.to(device)
    lit_model.eval()

    # Step 1: Choose one anchor sample from the batch
    import random
    anchor_idx = random.randint(0, len(view0) - 1)
    # anchor_idx = 0


    # Step 2: Get raw dataset and tokenizer
    original_dataset = train_loader.dataset.dataset

    # Step 3: Get the exon name for the anchor sample
    exon_name = original_dataset.exon_names[anchor_idx]
    all_views_dict = original_dataset.data[exon_name]  # species → seq

    # Step 4: Tokenize all augmentations of that anchor
    augmentations = list(all_views_dict.values())
    aug_tensor = torch.stack([
        torch.tensor(tokenizer(seq)["input_ids"]) for seq in augmentations
    ]).to(device)

    # Step 6: Get anchor embeddings for all other batch samples
    other_indices = [i for i in range(len(view0)) if i != anchor_idx]
    view0_others = view0[other_indices].to(device)
    lit_model.eval()
    with torch.no_grad():
        z_anchor_aug = lit_model(aug_tensor)  # (N_species, D)

        # Step 6: Get anchor embeddings for all other batch samples
        other_indices = [i for i in range(len(view0)) if i != anchor_idx]
        view0_others = view0[other_indices].to(device)
        z_others = lit_model(view0_others)  # (B-1, D)


    # Step 7: Concatenate for t-SNE
    embeddings = torch.cat([z_anchor_aug, z_others], dim=0).cpu().numpy()

    # Step 8: Label: 0 = all views of anchor, 1 = other anchors
    labels = [0] * z_anchor_aug.shape[0] + [1] * z_others.shape[0]

    emb_2d = TSNE(n_components=2, perplexity=20).fit_transform(embeddings)

    # Step 10: Plot
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=emb_2d[:, 0], y=emb_2d[:, 1], hue=labels, palette=["red", "gray"], s=30)
    plt.title("t-SNE: All Augmentations of One Anchor vs. Others")
    plt.legend(labels=["Augmentations of One Exon", "Other Anchors"])
    plt.axis("off")
    plt.tight_layout()
    plt.title(f'id{anchor_idx}__{exon_name}')
    plt.savefig(f'/gpfs/commons/home/atalukder/Contrastive_Learning/code/ML_model/figures/tsne{trimester}.png')

def distance_to_pos_and_neg(config, device, view0, train_loader, tokenizer):
    # Load the pretrained model
    lit_model = create_lit_model(config)
    simclr_ckpt = "/gpfs/commons/home/atalukder/Contrastive_Learning/files/results/exprmnt_2025_04_05__22_51_31/weights/checkpoints/introns_cl/ResNet1D/199/best-checkpoint.ckpt"
    ckpt = torch.load(simclr_ckpt, map_location=device)
    state_dict = ckpt["state_dict"]
    cleaned_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    lit_model.load_state_dict(cleaned_state_dict, strict=False)
    lit_model.to(device)
    lit_model.eval()

    # Step 1: Choose a random anchor from the batch
    anchor_idx = random.randint(0, len(view0) - 1)

    # Step 2: Get the full dataset
    original_dataset = train_loader.dataset.dataset
    exon_name = original_dataset.exon_names[anchor_idx]
    all_views_dict = original_dataset.data[exon_name]

    # Step 3: Tokenize all augmentations of the anchor exon
    augmentations = list(all_views_dict.values())
    aug_tensor = torch.stack([
        torch.tensor(tokenizer(seq)["input_ids"]) for seq in augmentations
    ]).to(device)

    with torch.no_grad():
        # Get embeddings of the anchor's positive augmentations
        z_anchor_aug = lit_model(aug_tensor)  # (N_species, D)

        # Use the mean or first as representative anchor vector
        anchor_vec = z_anchor_aug.mean(dim=0)

        # Get embeddings for all other batch items (negatives)
        other_indices = [i for i in range(len(view0)) if i != anchor_idx]
        view0_others = view0[other_indices].to(device)
        z_others = lit_model(view0_others)

    # Step 4: Compute distances
    dist_to_pos = F.pairwise_distance(anchor_vec.unsqueeze(0), z_anchor_aug)  # (N_species,)
    dist_to_neg = F.pairwise_distance(anchor_vec.unsqueeze(0), z_others) 
    print()

@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(config: OmegaConf):

    # Register Hydra resolvers
    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
    OmegaConf.register_new_resolver('min', lambda x, y: min([x, y]))
    OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
    OmegaConf.register_new_resolver('optimal_workers', lambda: get_optimal_num_workers())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print and process configuration
    print_config(config, resolve=True)

    # Initialize the IntronsDataModule with dataset-specific configs
    data_module = ContrastiveIntronsDataModule(config
    )
    data_module.prepare_data()
    data_module.setup()

    train_loader = data_module.train_dataloader()
    tokenizer = data_module.tokenizer
    batch = next(iter(train_loader))  # one batch
    view0, view1 = batch  # assuming contrastive pair

    # get_2view_embedding(config, device, view0, view1)
    # all_pos_of_anchor(config, device, view0, train_loader, tokenizer)
    distance_to_pos_and_neg(config, device, view0, train_loader, tokenizer)
    

if __name__ == "__main__":
    main()
