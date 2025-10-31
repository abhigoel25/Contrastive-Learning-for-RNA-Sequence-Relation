import pickle
import os
import math
import argparse

def get_exons_with_few_introns(datafilepath):
    exons_with_few_introns = set()  # Set to store exons with 0 or 1 intron
    
    with open(datafilepath, 'rb') as merged_intron_seq_file:
        merged_data = pickle.load(merged_intron_seq_file)
        print(f"There are {len(merged_data)} exons in the file {datafilepath}")
        
        for exon, introns in merged_data.items():
            total_introns = len(introns)
            if total_introns <= 1:
                exons_with_few_introns.add(exon)
        
    print(f"Found {len(exons_with_few_introns)} exon(s) with 0 or 1 intron.")
    return exons_with_few_introns

def merge_and_save_exon_data(data_dir, file_names, exons_to_remove):
    merged_data = {}
    all_exon_names = set()

    # Load each file, merge contents, and skip exons in exons_to_remove
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            # Remove exons found in step 1
            data = {exon: introns for exon, introns in data.items() if exon not in exons_to_remove}
            merged_data.update(data)
            all_exon_names.update(data.keys())

    # Save the merged data into a new .pkl file
    output_pkl_path = os.path.join(data_dir, 'merged_intron_sequences.pkl')
    with open(output_pkl_path, 'wb') as output_file:
        pickle.dump(merged_data, output_file)

    print("Merging complete. The merged data is saved to 'merged_intron_sequences.pkl'")
    return output_pkl_path, all_exon_names

def generate_all_exon_names(output_txt_filepath, exon_names):
    with open(output_txt_filepath, 'w') as txt_file:
        for exon_name in sorted(exon_names):
            txt_file.write(f"{exon_name}\n")
    print(f"Exon names saved to {output_txt_filepath}")

def main(data_dir, file_names):
    # Step 1: Identify exons with 0 or 1 intron in each file
    exons_with_few_introns = set()
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        exons_with_few_introns.update(get_exons_with_few_introns(file_path))

    # Step 2: Merge all files, excluding exons with 0 or 1 intron
    output_pkl_path, final_exon_names = merge_and_save_exon_data(data_dir, file_names, exons_with_few_introns)

    # Step 3: Generate text file with all exon names in the final merged .pkl file
    output_txt_path = os.path.join(data_dir, 'all_exon_names.txt')
    generate_all_exon_names(output_txt_path, final_exon_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process exon and intron data files.")
    parser.add_argument("data_dir", type=str, help="Directory containing the data files")
    parser.add_argument("file_names", type=str, nargs='+', help="List of data file names")

    args = parser.parse_args()
    main(args.data_dir, args.file_names)
