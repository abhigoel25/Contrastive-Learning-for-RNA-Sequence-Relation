from pathlib import Path
from omegaconf import OmegaConf
import os

def find_contrastive_root(start: Path = Path(__file__)) -> Path:
    for parent in start.resolve().parents:
        if parent.name == "Contrastive_Learning":
            return parent
    raise RuntimeError("Could not find 'Contrastive_Learning' directory.")

# Set env var *before* hydra loads config
os.environ["CONTRASTIVE_ROOT"] = str(find_contrastive_root())
CONTRASTIVE_ROOT = find_contrastive_root()



import sys
import os
import time
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from omegaconf import OmegaConf
import hydra
import time


trimester = time.strftime("_%Y_%m_%d__%H_%M_%S")

# os.environ['WANDB_INIT_TIMEOUT'] = '600'
def get_optimal_num_workers():
    num_cpus = os.cpu_count()
    num_gpus = torch.cuda.device_count()
    return min(num_cpus // max(1, num_gpus), 8)


# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import print_config
from src.model.lit import create_lit_model
from src.datasets.lit import ContrastiveIntronsDataModule


######### parameters #############
result_dir = "exprmnt_2025_09_23__00_38_41"
######### parameters ##############
# exprmnt_2025_05_04__11_29_05

def get_best_checkpoint(config):
    # simclr_ckpt = f"{root_path}/files/results/{result_dir}/weights/checkpoints/introns_cl/{config.embedder._name_}/199/best-checkpoint.ckpt"
    return f"{str(CONTRASTIVE_ROOT)}/files/results/{result_dir}/weights/checkpoints/introns_cl/{config.embedder._name_}/{config.dataset.seq_len}/best-checkpoint.ckpt"
    # return str(CONTRASTIVE_ROOT / "files/results/exprmnt_2025_05_04__11_29_05/weights/checkpoints/introns_cl/ResNet1D/199/best-checkpoint.ckpt")


def get_config_path():
    # simclr_ckpt = f"{root_path}/files/results/{result_dir}/weights/checkpoints/introns_cl/{config.embedder._name_}/199/best-checkpoint.ckpt"
    return f"{str(CONTRASTIVE_ROOT)}/files/results/{result_dir}/files/configs/"
    # return str(CONTRASTIVE_ROOT / "files/results/exprmnt_2025_05_04__11_29_05/weights/checkpoints/introns_cl/ResNet1D/199/best-checkpoint.ckpt")



def load_pretrained_model(config, device):
    model = create_lit_model(config)
    ckpt = torch.load(get_best_checkpoint(config), map_location=device)
    state_dict = ckpt["state_dict"]
    cleaned_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model.model.encoder

def get_tsne_embedding(z0, z1):

    tsne = TSNE(n_components=2, perplexity=30)
    z0_2d = tsne.fit_transform(z0.cpu().numpy())
    z1_2d = tsne.fit_transform(z1.cpu().numpy())

    return z0_2d, z1_2d


def get_2view_embedding(config, device, view0, view1):
    model = load_pretrained_model(config, device)
    with torch.no_grad():
        if config.embedder._name_ == "MTSplice":
            z0 = model(view0[0].to(device), view0[1].to(device))
            z1 = model(view1[0].to(device), view1[1].to(device))
        else:
            z0 = model(view0.to(device))
            z1 = model(view1.to(device))

    # embeddings = torch.cat([z0, z1], dim=0).cpu().numpy()
    # tsne = TSNE(n_components=2, perplexity=30)
    # emb_2d = tsne.fit_transform(embeddings)

    # z0_2d = emb_2d[:z0.shape[0]]
    # z1_2d = emb_2d[z0.shape[0]:]

    # tsne = TSNE(n_components=2, perplexity=30)
    # z0_2d = tsne.fit_transform(z0.cpu().numpy())
    # z1_2d = tsne.fit_transform(z1.cpu().numpy())
    z0_2d, z1_2d = get_tsne_embedding(z0, z1)

    plt.figure(figsize=(8, 8))
    for i in range(z0.shape[0]):
        plt.plot([z0_2d[i, 0], z1_2d[i, 0]], [z0_2d[i, 1], z1_2d[i, 1]], color="gray", linewidth=0.5, alpha=0.5)
    plt.scatter(z0_2d[:, 0], z0_2d[:, 1], color="blue", s=20, label="Anchor (z0)")
    plt.scatter(z1_2d[:, 0], z1_2d[:, 1], color="orange", s=20, label="Positive (z1)")
    plt.legend()
    plt.title("t-SNE: Anchor–Positive Pairs")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f'{main_dir}/figures/tsne{time.strftime("_%Y_%m_%d__%H_%M_%S")}.png')

    # plt.savefig(f'../figures/tsne{time.strftime("_%Y_%m_%d__%H_%M_%S")}.png')




def all_pos_of_anchor(config, device, view0, train_loader, batch):
    
    model = load_pretrained_model(config, device)
    # anchor_idx = random.randint(0, len(view0) - 1)
    anchor_idx = 10
    # dataset = train_loader.dataset.dataset
    dataset = train_loader.dataset
    exon_name = dataset.exon_names[anchor_idx]
    all_views_dict = dataset.data[exon_name]

    augmentations = list(all_views_dict.values())
    

    if callable(tokenizer) and not hasattr(tokenizer, "vocab_size"):
        aug_tensor = torch.stack([
            tokenizer([seq])[0] for seq in augmentations
        ]).to(device)

    elif callable(tokenizer):  # HuggingFace-style
            aug_tensor = torch.stack([
            torch.tensor(tokenizer(seq)["input_ids"]) for seq in augmentations
        ]).to(device)
    else:
        print()

    other_indices = [i for i in range(len(view0)) if i != anchor_idx]
    view0_others = view0[other_indices].to(device)

    with torch.no_grad():
        z_anchor_aug = model(aug_tensor)
        z_others = model(view0_others)

    embeddings = torch.cat([z_anchor_aug, z_others], dim=0).cpu().numpy()
    labels = [0] * z_anchor_aug.shape[0] + [1] * z_others.shape[0]

    emb_2d = TSNE(n_components=2, perplexity=20).fit_transform(embeddings)

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=emb_2d[:, 0], y=emb_2d[:, 1], hue=labels, palette=["red", "gray"], s=30)
    plt.title(f"t-SNE: All Augmentations of One Anchor vs. Others\nid{anchor_idx}__{exon_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.title(f'id{anchor_idx}__{exon_name}')
    # plt.savefig(f'{main_dir}figures/all_pos_of_anchor{time.strftime("_%Y_%m_%d__%H_%M_%S")}.png')
    plt.savefig(f'{main_dir}/figures/all_pos_of_anchor{time.strftime("_%Y_%m_%d__%H_%M_%S")}.png')


def distance_to_pos_and_neg(config, device, view0, train_loader, tokenizer):
    model = load_pretrained_model(config, device)
    anchor_idx = random.randint(0, len(view0) - 1)
    dataset = train_loader.dataset.dataset
    exon_name = dataset.exon_names[anchor_idx]
    all_views_dict = dataset.data[exon_name]

    augmentations = list(all_views_dict.values())
    aug_tensor = torch.stack([
        torch.tensor(tokenizer(seq)["input_ids"]) for seq in augmentations
    ]).to(device)

    other_indices = [i for i in range(len(view0)) if i != anchor_idx]
    view0_others = view0[other_indices].to(device)

    with torch.no_grad():
        z_anchor_aug = model(aug_tensor)
        anchor_vec = z_anchor_aug.mean(dim=0)
        z_others = model(view0_others)

    dist_to_pos = F.pairwise_distance(anchor_vec.unsqueeze(0), z_anchor_aug)
    dist_to_neg = F.pairwise_distance(anchor_vec.unsqueeze(0), z_others)

    print(f"Anchor: {exon_name} (idx {anchor_idx})")
    print(f"Distances to positives (mean): {dist_to_pos.mean():.4f}")
    print(f"Distances to negatives (mean): {dist_to_neg.mean():.4f}")



# Define main directory for code files (e.g., for saving plots)
main_dir = str(CONTRASTIVE_ROOT / "code" / "ML_model")


@hydra.main(version_base=None, config_path=get_config_path(), config_name="config.yaml")
def main(config: OmegaConf):
    # Register Hydra resolvers
    # OmegaConf.register_new_resolver("contrastive_root", lambda: str(CONTRASTIVE_ROOT))
    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
    OmegaConf.register_new_resolver('min', lambda x, y: min([x, y]))
    OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
    OmegaConf.register_new_resolver('optimal_workers', lambda: get_optimal_num_workers())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_config(config, resolve=True)
   
    data_module = ContrastiveIntronsDataModule(config)
    data_module.prepare_data()
    data_module.setup()

    train_loader = data_module.train_dataloader()
    # tokenizer = data_module.tokenizer
    # view0, view1, _, _ = next(iter(train_loader))
    batch = next(iter(train_loader))

    # Choose one of the following:
    view0, view1 = batch[0], batch[1]
    # get_2view_embedding(config, device, view0, view1)
    all_pos_of_anchor(config, device, view0, train_loader, batch) # choses 1 exon and plots all its augmentations against other exons's anchor view
    # distance_to_pos_and_neg(config, device, view0, train_loader, tokenizer)

if __name__ == "__main__":
    main()
