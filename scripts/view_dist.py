import pickle
from collections import Counter
import matplotlib.pyplot as plt

# Path to your pkl file
train_pkl = "/mnt/home/at3836/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/train_3primeIntron_filtered.pkl"



import pickle

# Path to input and output files
input_pkl = "/mnt/home/at3836/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/train_3primeIntron_filtered.pkl"
output_pkl = "/mnt/home/at3836/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/train_3primeIntron_filtered_min30views.pkl"


# Load data
with open(input_pkl, 'rb') as f:
    data = pickle.load(f)

# Filter to keep only exons with >= 30 augmentations/views
filtered_data = {exon: species_dict for exon, species_dict in data.items() if len(species_dict) >= 30}

print(f"Original exons: {len(data)}")
print(f"Exons with at least 30 views: {len(filtered_data)}")

# Optionally, save to a new file
with open(output_pkl, 'wb') as f:
    pickle.dump(filtered_data, f)

print(f"Saved filtered data to: {output_pkl}")







# # Load data
# with open(train_pkl, 'rb') as f:
#     data = pickle.load(f)

# # Count number of views (augmentations/species) for each exon
# view_counts = [len(species_dict) for species_dict in data.values()]

# # Summary statistics
# print("Total exons:", len(view_counts))
# print("Min views:", min(view_counts))
# print("Max views:", max(view_counts))
# print("Mean views:", sum(view_counts)/len(view_counts))
# print("View count distribution:")
# print(Counter(view_counts))

# # Optionally, plot a histogram
# plt.hist(view_counts, bins=range(min(view_counts), max(view_counts)+2), edgecolor='black')
# plt.xlabel('Number of augmentations/views per exon')
# plt.ylabel('Number of exons')
# plt.title('Distribution of views/augmentations in training set')
# plt.savefig("dis.png")
# plt.show()
