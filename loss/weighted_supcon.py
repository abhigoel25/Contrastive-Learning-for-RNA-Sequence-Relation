"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import pickle
import pandas as pd

class weightedSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='one',
                 base_temperature=0.07, correlation_dir=None):
        super(weightedSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.correlation_dir = correlation_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # --- Create the master name-to-index map FIRST ---
        # Load the master list of all unique exon names you just created
        master_exon_csv_path = f"{correlation_dir}/all_exon_list.csv"
        master_df = pd.read_csv(master_exon_csv_path)
        all_exon_names = master_df['exon_id'].tolist()
        
        # This map is now universal for all splits and is created only once
        name_to_global_idx = {name: i for i, name in enumerate(all_exon_names)}
        self.name_to_global_idx = name_to_global_idx
        num_total_exons = len(all_exon_names)
        

        # Dictionaries to hold the data for each split
        self.mad_matrix_tensors = {}
        self.global_d_scores = {}
        self.global_to_alt_idx = {}
        self.is_alternate_masks = {}

        # --- Loop through each division and load its data ---
        for division in ['train', 'val']:
            print(f"Loading correlation data for: {division}")

            weight_matrix_path = f"{correlation_dir}/{division}_ExonExon_meanAbsDist.pkl"
            with open(weight_matrix_path, "rb") as f:
                weight_matrix_df = pickle.load(f)
            
            # name_to_global_idx = {name: i for i, name in enumerate(all_exon_names)} #given each exon name an index
            num_total_exons = len(all_exon_names)
            alt_exon_names = weight_matrix_df.index.tolist() # a list of all alternate exon names in the current division from weight matrix

            mad_tensor = torch.from_numpy(
                weight_matrix_df.drop(columns=['D_score']).values
            ).float() # drops the D_score column and converts the rest to a tensor
            alt_d_scores = torch.from_numpy(weight_matrix_df['D_score'].values).float()

            global_d = torch.zeros(num_total_exons)
            alt_global_indices = torch.tensor([name_to_global_idx.get(name) for name in alt_exon_names])
            global_d[alt_global_indices] = alt_d_scores #global d is the MAD to constitutive exons from alternating exons arranged as alt index 
            
            # Create a lookup tensor that translates from a global index (~800k) to a local index (~29k).
            # This is used to find the correct row/column in the smaller `mad_matrix_tensor`
            # for any given alternate exon using its universal global ID.
            # - The INDEX of this tensor corresponds to the GLOBAL exon index.
            # - The VALUE at that index provides the LOCAL index for the MAD matrix.
            # - A value of -1 indicates a constitutive exon, which has no entry in the MAD matrix.
            global_to_alt = torch.full((num_total_exons,), -1, dtype=torch.long)
            global_to_alt[alt_global_indices] = torch.arange(len(alt_exon_names))
            
            is_alt_mask = torch.zeros(num_total_exons, dtype=torch.bool)
            is_alt_mask[alt_global_indices] = True

            self.mad_matrix_tensors[division] = mad_tensor.to(self.device)
            self.global_d_scores[division] = global_d.to(self.device)
            self.global_to_alt_idx[division] = global_to_alt.to(self.device)
            self.is_alternate_masks[division] = is_alt_mask.to(self.device)

    def forward(self, features, exon_name, division,labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device) # identity matrix batch_size x batch_size
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # (batchsize . view) x dimension
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        """
        (AT)
        When you use torch.matmul(anchor_feature, contrast_feature.T) without normalizing the rows of anchor_feature and contrast_feature, 
        The dot product will be proportional to the magnitude (norm) of each vector. If your features are not L2 normalized, you can easily get huge numbers (hundreds or thousands).
        """
        anchor_feature = nn.functional.normalize(anchor_feature, dim=1)
        contrast_feature = nn.functional.normalize(contrast_feature, dim=1)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # # compute log_prob
        # exp_logits = torch.exp(logits) * logits_mask
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        ################################################################
        # MODIFICATION FOR WEIGHTED LOSS STARTS HERE
        ################################################################

        # corr_file_path = f"{self.correlation_dir}/{division}_ExonExon_meanAbsDist.pkl"
        # with open(corr_file_path, "rb") as f:
        #     self.correlation_df = pickle.load(f)

        # # 2. For the current batch, get the corresponding correlation weights
        # # Expand exon names to match the anchor and contrast dimensions
        # # e.g., if batch_size=4, anchor_count=2 -> ['e1','e2','e3','e4','e1','e2','e3','e4']
        # anchor_names = exon_name * anchor_count
        # contrast_names = exon_name * contrast_count

        # # Efficiently grab the sub-matrix of correlations using pandas
        # # This creates a matrix where W[i, j] is the correlation between
        # # anchor_name[i] and contrast_name[j].
        # correlation_submatrix = self.correlation_df.loc[anchor_names, contrast_names]
        
        # # Convert to a PyTorch tensor and move to the correct device
        # weights = torch.from_numpy(correlation_submatrix.values).float().to(device)
        # weights = torch.nan_to_num(weights, nan=0.0)

        # # Compute log_prob with the weights applied to the denominator
        # exp_logits = torch.exp(logits) * logits_mask
        
        # # 3. Apply the weights before summing to get the new denominator
        # weighted_sum_exp_logits = (weights * exp_logits).sum(1, keepdim=True)
        
        # # Add a small epsilon for numerical stability to avoid log(0)
        # log_prob = logits - torch.log(weighted_sum_exp_logits + 1e-9)

        # # --- In your forward pass / loss function ---

        # # 1. Convert batch names to their global integer indices
        # anchor_names = exon_name * anchor_count
        # contrast_names = exon_name * contrast_count

        # anchor_global_indices = torch.tensor([self.name_to_global_idx[name] for name in anchor_names], device=device, dtype=torch.long)
        # contrast_global_indices = torch.tensor([self.name_to_global_idx[name] for name in contrast_names], device=device, dtype=torch.long)

        # # 2. Get boolean masks for exon types using the pre-built global mask
        # is_anchor_alternate = self.is_alternate_mask[anchor_global_indices]
        # is_contrast_alternate = self.is_alternate_mask[contrast_global_indices]
        # is_anchor_constitutive = ~is_anchor_alternate
        # is_contrast_constitutive = ~is_contrast_alternate

        # # 3. Initialize the final weight matrix
        # weights = torch.zeros_like(logits, device=device)

        # # --- Case 2 & 3: Anchor/Contrast pairs involving one constitutive exon ---
        # # Gather the D-scores for all anchors and contrasts in the batch from the global lookup
        # d_scores_anchors = self.global_d_scores[anchor_global_indices]
        # d_scores_contrasts = self.global_d_scores[contrast_global_indices]

        # # Create masks for the two cases
        # mask_alt_anchor_const_contrast = is_anchor_alternate.unsqueeze(1) & is_contrast_constitutive.unsqueeze(0)
        # mask_const_anchor_alt_contrast = is_anchor_constitutive.unsqueeze(1) & is_contrast_alternate.unsqueeze(0)

        # # Apply weights using broadcasting. PyTorch will correctly expand the vectors.
        # weights += mask_alt_anchor_const_contrast * d_scores_anchors.unsqueeze(1)
        # weights += mask_const_anchor_alt_contrast * d_scores_contrasts.unsqueeze(0)

        # # --- Case 1: Anchor Alternate, Contrast Alternate ---
        # # Find the locations for alt-alt pairs
        # mask_alt_alt = is_anchor_alternate.unsqueeze(1) & is_contrast_alternate.unsqueeze(0)

        # # Only proceed if there are any alt-alt pairs to avoid empty tensor errors
        # if torch.any(mask_alt_alt):
        #     # Map from the batch's global indices to the local indices of the MAD matrix
        #     anchor_local_indices = self.global_to_alt_idx[anchor_global_indices]
        #     contrast_local_indices = self.global_to_alt_idx[contrast_global_indices]

        #     # Use meshgrid and advanced indexing to gather the MAD values for the entire batch at once
        #     # We only need the indices where the anchor is alternate
        #     ii = anchor_local_indices[is_anchor_alternate].view(-1, 1)
        #     jj = contrast_local_indices[is_contrast_alternate].view(1, -1)
            
        #     mad_values = self.mad_matrix_tensor[ii, jj]

        #     # Place the gathered MAD values into the final weights matrix using the alt-alt mask
        #     weights[mask_alt_alt] = mad_values.flatten()


        # # --- The rest of your code remains the same ---
        # weights = torch.nan_to_num(weights, nan=0.0)
        # exp_logits = torch.exp(logits) * logits_mask
        # weighted_sum_exp_logits = (weights * exp_logits).sum(1, keepdim=True)
        # log_prob = logits - torch.log(weighted_sum_exp_logits + 1e-9)

        # FIX 5: Select the correct tensors for the current division
        mad_matrix = self.mad_matrix_tensors[division].to(device)
        global_d_scores = self.global_d_scores[division].to(device)
        global_to_alt_idx = self.global_to_alt_idx[division].to(device)
        is_alternate_mask = self.is_alternate_masks[division].to(device)

        # 1. Convert batch names to their global integer indices
        anchor_names = [name for name in exon_name for _ in range(anchor_count)]
        contrast_names = [name for name in exon_name for _ in range(contrast_count)]
        
        anchor_global_indices = torch.tensor([self.name_to_global_idx[name] for name in anchor_names], device=device, dtype=torch.long)
        contrast_global_indices = torch.tensor([self.name_to_global_idx[name] for name in contrast_names], device=device, dtype=torch.long)

        # 2. Get boolean masks for exon types
        is_anchor_alternate = is_alternate_mask[anchor_global_indices]
        is_contrast_alternate = is_alternate_mask[contrast_global_indices]
        
        # 3. Build weight matrix
        weights = torch.zeros_like(logits, device=device)
        d_scores_anchors = global_d_scores[anchor_global_indices]
        d_scores_contrasts = global_d_scores[contrast_global_indices]

        mask_alt_anchor_const_contrast = is_anchor_alternate.unsqueeze(1) & ~is_contrast_alternate.unsqueeze(0) # if anchor is alt and contrast is constitutive
        mask_const_anchor_alt_contrast = ~is_anchor_alternate.unsqueeze(1) & is_contrast_alternate.unsqueeze(0) # if anchor is constitutive and contrast is alt

        weights += mask_alt_anchor_const_contrast * d_scores_anchors.unsqueeze(1)
        weights += mask_const_anchor_alt_contrast * d_scores_contrasts.unsqueeze(0)
        
        mask_alt_alt = is_anchor_alternate.unsqueeze(1) & is_contrast_alternate.unsqueeze(0)

        if torch.any(mask_alt_alt):
            anchor_local_indices = global_to_alt_idx[anchor_global_indices[is_anchor_alternate]]
            contrast_local_indices = global_to_alt_idx[contrast_global_indices[is_contrast_alternate]]
            
            mad_values = mad_matrix[anchor_local_indices[:, None], contrast_local_indices]
            # weights[mask_alt_alt] = mad_values.flatten() if len(mad_values.shape) > 0 else mad_values
            weights[mask_alt_alt] = mad_values.to(weights.dtype).flatten() if len(mad_values.shape) > 0 else mad_values.to(weights.dtype)


        # Compute log_prob with weights
        weights = torch.nan_to_num(weights, nan=0.0)
        exp_logits = torch.exp(logits) * logits_mask
        weighted_sum_exp_logits = (weights * exp_logits).sum(1, keepdim=True)
        log_prob = logits - torch.log(weighted_sum_exp_logits + 1e-9)
        
        ################################################################
        # MODIFICATION ENDS HERE
        ################################################################

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. S
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        # print(f"🦀 weightedSupConLoss: {loss.item():.4f}")

        return loss