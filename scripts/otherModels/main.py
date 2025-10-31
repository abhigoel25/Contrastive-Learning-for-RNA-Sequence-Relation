
# from visualize.tsne_plot import plot_2view_tsne, plot_2view_tsne_numpy

embedder_name = "borzoi"  # or "enformer"

if embedder_name == "borzoi":
    from embedder.borzoi_embedder import BorzoiEmbedder
    model_path = "/home/atalukder/Contrastive_Learning/models/borzoi/examples/saved_models/f3c0/train/model0_best.h5"
    embedder = BorzoiEmbedder(model_path, seq_len=524288)

    # 2 dummy sequences
    seqs = ["A" * 200]
    embeddings = embedder.embed(seqs)

    print(embeddings.shape)

    # plot_2view_tsne_numpy(z0, z1, save_path="tsne_borzoi.png")

elif embedder_name == "enformer":
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from embedder.enformer_embedder import EnformerEmbedder
    encoder = EnformerEmbedder(device)

    # Fake 1-hot PyTorch input
    view0 = torch.randn(16, 1000, 4)
    view1 = torch.randn(16, 1000, 4)

    plot_2view_tsne(encoder, view0, view1, save_path="tsne_enformer.png")

else:
    raise ValueError("Unknown embedder")
