# import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np

def plot_2view_tsne(encoder, view0, view1, save_path):
    with torch.no_grad():
        z0 = encoder.embed(view0)
        z1 = encoder.embed(view1)

    embeddings = torch.cat([z0, z1], dim=0).cpu().numpy()
    emb_2d = TSNE(n_components=2, perplexity=30).fit_transform(embeddings)

    z0_2d, z1_2d = emb_2d[:len(z0)], emb_2d[len(z0):]

    plt.figure(figsize=(8, 8))
    for i in range(len(z0)):
        plt.plot([z0_2d[i, 0], z1_2d[i, 0]], [z0_2d[i, 1], z1_2d[i, 1]], color="gray", alpha=0.5)
    plt.scatter(z0_2d[:, 0], z0_2d[:, 1], color="blue", label="Anchor")
    plt.scatter(z1_2d[:, 0], z1_2d[:, 1], color="orange", label="Positive")
    plt.legend()
    plt.axis("off")
    plt.title("t-SNE: Anchor-Positive Pairs")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()