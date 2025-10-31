import pickle

path = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/ASCOT_finetuning/psi_val_Retina___Eye_psi_MERGED.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)

print()