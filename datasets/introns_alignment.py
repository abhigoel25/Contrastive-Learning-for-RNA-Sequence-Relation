import os
import numpy as np
import torch
import random
import pickle
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from .utility import (
    get_windows_with_padding,
    get_windows_with_padding_intronOnly,
)
############# DEBUG Message ###############
import inspect
import os
_warned_debug = False  # module-level flag
def reset_debug_warning():
    global _warned_debug
    _warned_debug = False
def debug_warning(message):
    global _warned_debug
    if not _warned_debug:
        frame = inspect.currentframe().f_back
        filename = os.path.basename(frame.f_code.co_filename)
        lineno = frame.f_lineno
        print(f"\033[1;31m⚠️⚠️ ⚠️ ⚠️ DEBUG MODE ENABLED in {filename}:{lineno} —{message} REMEMBER TO REVERT!\033[0m")
        _warned_debug = True
############# DEBUG Message ###############

class ContrastiveIntronsDataset(Dataset):
    def __init__(self, data_file, n_augmentations=2, embedder=None, fixed_species=0):
        # Load the merged data and exon names
        with open(data_file, 'rb') as file:
            self.data = pickle.load(file)
        self.exon_names = list(self.data.keys())  
        self.exon_name_to_id = {name: i for i, name in enumerate(self.exon_names)}
        self.n_augmentations = n_augmentations   # <-- store as an attribute
        self.embedder = embedder
        self.embedder_name = embedder._name_ 
        # Fixed lengths for MTSplice windowing
        self.len_5p = 200
        self.len_3p = 200

        # reset_debug_warning()
        # debug_warning("no exon, so acceptor, donor intron is 400, generally 300.")
        # reset_debug_warning()
        # debug_warning("get padding intronlyONLY, line 151")
        
        self.tissue_acceptor_intron = 300
        self.tissue_donor_intron = 300
        
        self.tissue_acceptor_exon = 100
        self.tissue_donor_exon = 100

        self.fixed_species = fixed_species

        if fixed_species:
            # --- NEW: choose a fixed species list once (same order for every sample) ---
            from collections import Counter
            from itertools import chain

            # Build per-exon species sets
            _species_sets = [set(self.data[ex].keys()) for ex in self.exon_names]
            _common = set.intersection(*_species_sets) if _species_sets else set()

            if isinstance(self.n_augmentations, str) and str(self.n_augmentations).lower() == "all":
                # fall back to top-K using the most common species overall
                counts = Counter(chain.from_iterable(self.data[ex].keys() for ex in self.exon_names))
                self.fixed_species = [s for s, _ in counts.most_common(len(counts))]   # all, ordered by freq
                self.n_augmentations = len(self.fixed_species)
            else:
                if len(_common) >= int(self.n_augmentations):
                    self.fixed_species = sorted(list(_common))[:int(self.n_augmentations)]
                    # self.fixed_species = random.sample(list(_common), int(self.n_augmentations))
                else:
                    counts = Counter(chain.from_iterable(self.data[ex].keys() for ex in self.exon_names))
                    self.fixed_species = [s for s, _ in counts.most_common(int(self.n_augmentations))]
                    # species_sorted = [s for s, _ in counts.most_common()]  # all species by freq
                    # random.shuffle(species_sorted)                          # randomize once
                    # self.fixed_species = species_sorted[:int(self.n_augmentations)]

            # Keep only exons that contain all fixed species (uniform views)
            before = len(self.exon_names)
            self.exon_names = [ex for ex in self.exon_names
                            if set(self.fixed_species).issubset(self.data[ex].keys())]
            self.exon_name_to_id = {name: i for i, name in enumerate(self.exon_names)}
            
            # Build a list of indices whose exons have all fixed species
            self._valid_idx = [i for i, ex in enumerate(self.exon_names)
                    if set(self.fixed_species).issubset(self.data[ex].keys())]
            print(f"[Dataset] Fixed species: {self.fixed_species} | kept {len(self._valid_idx)}/{before} exons")


            # --- END NEW ---

        # Fixed seed
        # debug_warning() 
        # self.seed = 42
        #  # will only print the first time
        # self.random_state = random.Random(self.seed)
 
    
    def __len__(self):

        if self.fixed_species:
            return len(self._valid_idx)
        return len(self.data)
        

    # import random
    # reset_debug_warning()
    # debug_warning("getitem")

    # import random

    # def __getitem__(self, idx):

    #     # ==========================================================================
    #     # --- DEBUG BLOCK: Force selection from a fixed list of 4 exons ---
    #     debug_exon_names = [
    #         'ENST00000275517.8_4_6', 
    #         'ENST00000423302.7_7_21', 
    #         'ENST00000003084.11_1_27', 
    #         'ENST00000003302.8_5_25'
    #     ]
        
    #     # Use the index to cycle through the four names.
    #     # idx=0 -> name 1, idx=1 -> name 2, ..., idx=4 -> name 1 again
    #     exon_name = debug_exon_names[idx % len(debug_exon_names)]
        
    #     # --- End of DEBUG BLOCK ---
    #     # ==========================================================================

    

    #     # --- FIXED: use the same species for every sample ---
    #     if hasattr(self, "fixed_species") and self.fixed_species:
    #         real_i = self._valid_idx[idx]
    #         exon_name = self.exon_names[real_i]
    #         intronic_sequences = self.data[exon_name]
    #         exon_id = self.exon_name_to_id[exon_name]
    #         species_sample = self.fixed_species
    #         # sanity (should never fail if you filtered in __init__)
    #         missing = [sp for sp in species_sample if sp not in intronic_sequences]
    #         if missing:
    #             raise KeyError(f"Exon {exon_name} missing fixed species: {missing}")
    #     else:
    #         # Now the rest of the code will use the correct exon_name from the debug block
    #         exon_id = self.exon_name_to_id[exon_name]
    #         intronic_sequences = self.data[exon_name]
    #         all_species = list(intronic_sequences.keys())
    #         n_available = len(all_species)
    #         if self.n_augmentations == "all" or self.n_augmentations > n_available:
    #             species_sample = all_species
    #         else:
    #             species_sample = random.sample(all_species, self.n_augmentations)

    #     if self.embedder_name == "MTSplice":
    #         augmentations = []
    #         for sp in species_sample:
    #             intron_5p = intronic_sequences[sp]['5p']
    #             exon_seq = intronic_sequences[sp]['exon']
    #             intron_3p = intronic_sequences[sp]['3p']

    #             len_exon = len(exon_seq)
    #             # full_seq = intron_5p + exon_seq + intron_3p
    #             full_seq = intron_5p + exon_seq + intron_3p
                
    #             # full_seq = intronic_sequences[sp]

    #             if self.embedder.name_or_path == "MTSplice":
                    
    #                 windows = get_windows_with_padding(self.tissue_acceptor_intron, self.tissue_donor_intron, self.tissue_acceptor_exon, self.tissue_donor_exon, full_seq, overhang = (self.len_3p, self.len_5p))
                
    #                 # windows = self.get_windows_with_padding_intronOnly(
    #                 #     full_seq, overhang = (self.len_3p, self.len_5p))
                    

    #                 # augmentations.append((windows['acceptor'], windows['donor']))
    #                 augmentations.append({'acceptor': windows['acceptor'], 'donor': windows['donor']})

    #             else:
    #                 augmentations.append(full_seq)
    #     else:
    #         exon_name = self.exon_names[idx]
    #         exon_id = self.exon_name_to_id[exon_name]
    #         intronic_sequences = self.data[exon_name]

    #         # Number of available augmentations
    #         all_species = list(intronic_sequences.keys())
    #         n_available = len(all_species)

    #         # Retrieve the sequences for the sampled species
    #         augmentations = [intronic_sequences[sp] for sp in species_sample]


    #     return augmentations, exon_id, exon_name  # return exon_name for debugging

    

    def __getitem__(self, idx):
    

        # --- FIXED: use the same species for every sample ---
        if hasattr(self, "fixed_species") and self.fixed_species:
            real_i = self._valid_idx[idx]
            exon_name = self.exon_names[real_i]
            intronic_sequences = self.data[exon_name]
            exon_id = self.exon_name_to_id[exon_name]
            species_sample = self.fixed_species
            # sanity (should never fail if you filtered in __init__)
            missing = [sp for sp in species_sample if sp not in intronic_sequences]
            if missing:
                raise KeyError(f"Exon {exon_name} missing fixed species: {missing}")
        else:
            exon_name = self.exon_names[idx]
            exon_id = self.exon_name_to_id[exon_name]
            intronic_sequences = self.data[exon_name]
            exon_id = self.exon_name_to_id[exon_name]
            # original fallback behavior
            all_species = list(intronic_sequences.keys())
            n_available = len(all_species)
            if self.n_augmentations == "all" or self.n_augmentations > n_available:
                species_sample = all_species
            else:
                species_sample = random.sample(all_species, self.n_augmentations)

        if self.embedder_name == "MTSplice":
            augmentations = []
            for sp in species_sample:
                intron_5p = intronic_sequences[sp]['5p']
                exon_seq = intronic_sequences[sp]['exon']
                intron_3p = intronic_sequences[sp]['3p']

                len_exon = len(exon_seq)
                # full_seq = intron_5p + exon_seq + intron_3p
                full_seq = intron_5p + exon_seq + intron_3p
                
                # full_seq = intronic_sequences[sp]

                if self.embedder.name_or_path == "MTSplice":
                    
                    windows = get_windows_with_padding(self.tissue_acceptor_intron, self.tissue_donor_intron, self.tissue_acceptor_exon, self.tissue_donor_exon, full_seq, overhang = (self.len_3p, self.len_5p))
                
                    # windows = self.get_windows_with_padding_intronOnly(
                    #     full_seq, overhang = (self.len_3p, self.len_5p))
                    

                    # augmentations.append((windows['acceptor'], windows['donor']))
                    augmentations.append({'acceptor': windows['acceptor'], 'donor': windows['donor']})

                else:
                    augmentations.append(full_seq)
        else:
            exon_name = self.exon_names[idx]
            exon_id = self.exon_name_to_id[exon_name]
            intronic_sequences = self.data[exon_name]

            # Number of available augmentations
            all_species = list(intronic_sequences.keys())
            n_available = len(all_species)

            # Retrieve the sequences for the sampled species
            augmentations = [intronic_sequences[sp] for sp in species_sample]


        return augmentations, exon_id, exon_name  # return exon_name for debugging


