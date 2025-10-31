import pickle
import os
import math
import argparse

main_path = '/home/atalukder/mnt_NYGC/'
# datafilepath = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/fine_tuning/Psi_values/psi_Lung_exon_sequences_dict_50to1000bpWinf_nan_vals.pkl'
datafilepath = main_path+'data/fine_tuning/Psi_values/psi_Lung_exon_sequences_dict_50to1000bpWinf_nan_vals.pkl'

with open(datafilepath, 'rb') as merged_intron_seq_file:
        merged_data = pickle.load(merged_intron_seq_file)
        print()

print(f"initial dataset size: {len(merged_data)} entries")

nan_or_inf_count = sum(
    1 for v in merged_data.values()
    if math.isnan(v["psi_val"]) or math.isinf(v["psi_val"])
)

print(f"⚠️ Found {nan_or_inf_count} entries with NaN or inf PSI values.")


# Filter to keep only valid entries
merged_data = {
    k: v for k, v in merged_data.items()
    if not (math.isnan(v["psi_val"]) or math.isinf(v["psi_val"]))
}

print(f"✅ Cleaned dataset size: {len(merged_data)} entries")

nan_or_inf_count = sum(
    1 for v in merged_data.values()
    if math.isnan(v["psi_val"]) or math.isinf(v["psi_val"])
)

print(f"⚠️ Found {nan_or_inf_count} entries with NaN or inf PSI values.")


# output_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/fine_tuning/Psi_values/psi_Lung_intron_sequences_dict_50to1000bp.pkl'
output_path = main_path+'data/fine_tuning/Psi_values/psi_Lung_exon_sequences_dict_50to1000bp.pkl'

with open(output_path, "wb") as f:
    pickle.dump(merged_data, f)

print(f"📦 Cleaned data saved to: {output_path}")



# def main(data_dir, file_names):
#     # Step 1: Identify exons with 0 or 1 intron in each file
#     exons_with_few_introns = set()
#     for file_name in file_names:
#         file_path = os.path.join(data_dir, file_name)
#         exons_with_few_introns.update(get_exons_with_few_introns(file_path))

#     # Step 2: Merge all files, excluding exons with 0 or 1 intron
#     output_pkl_path, final_exon_names = merge_and_save_exon_data(data_dir, file_names, exons_with_few_introns)

#     # Step 3: Generate text file with all exon names in the final merged .pkl file
#     output_txt_path = os.path.join(data_dir, 'all_exon_names.txt')
#     generate_all_exon_names(output_txt_path, final_exon_names)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process exon and intron data files.")
#     parser.add_argument("data_dir", type=str, help="Directory containing the data files")
#     parser.add_argument("file_names", type=str, nargs='+', help="List of data file names")

#     args = parser.parse_args()
#     main(args.data_dir, args.file_names)
