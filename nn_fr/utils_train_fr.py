# utils_train_fr.py
import math
import random
try:
    import h5py
except Exception:
    h5py = None
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import PositionalEncoding, GraphNorm
from torch_geometric.utils import dense_to_sparse


# ------------------------------------------------------
# Losses
# ------------------------------------------------------
def get_loss_function(name: str):
    name = name.lower()
    if name == 'mse':
        return nn.MSELoss()
    if name == 'mae':
        return nn.L1Loss()
    if name == 'huber':
        return nn.HuberLoss()
    if name == 'smooth_l1':
        return nn.SmoothL1Loss()
    raise ValueError(f'Unknown loss_function: {name}')


# ------------------------------------------------------
# Dataset holder
# ------------------------------------------------------
class CustomGraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# ------------------------------------------------------
# Optional: KNN graph (if you don't want full graph)
# ------------------------------------------------------
def knn_graph_from_dist_matrix(dist_matrix: torch.Tensor, k: int) -> torch.Tensor:
    num_nodes = dist_matrix.size(0)
    edge_index = []
    for i in range(num_nodes):
        distances = dist_matrix[i].cpu().numpy()
        nearest_indices = np.argsort(distances)[1:k+1]  # exclude self
        for j in nearest_indices:
            edge_index.append([i, j])
            edge_index.append([j, i])  # undirected
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index


# ------------------------------------------------------
# f scalar (graph-level) from scaling + raw flying_range
# ------------------------------------------------------
# def compute_f_scalar(scale_factor: float, flying_range: float, mode: str = "squash") -> float:
#     """
#     Convert raw flying_range (same physical units as original distances)
#     to the normalized scalar f in [0,1]-ish using the same scale_factor
#     used when distances were scaled.

#       r_eff = flying_range / scale_factor
#       squash:  f = 1 / (1 + r_eff)
#       inverse: f = 0 if r_eff<=0 else 1/r_eff
#       exp:     f = exp(-r_eff)
#     """
#     if flying_range is None or math.isinf(float(flying_range)):
#         # print("Flying range is None or inf")
#         # print("f is 0.0")
#         return 0.0
#     sf = float(scale_factor) if (scale_factor is not None and scale_factor > 0) else 1.0
#     r_eff = max(0.0, float(flying_range) / sf)

#     if mode == "squash":
#         return 1.0 / (1.0 + r_eff)
#     elif mode == "inverse":
#         return 0.0 if r_eff <= 0.0 else 1.0 / r_eff
#     elif mode == "exp":
#         return math.exp(-r_eff)
#     else:
#         raise ValueError(f"Unknown f mode: {mode}")

def compute_f_from_percentage(flying_range_percent: float) -> float:
    """
    Converts a flying range percentage into a scalar feature 'f'.
    Handles the infinite case for identity checks.
    """
    if flying_range_percent is None or math.isinf(float(flying_range_percent)):
        return 0.0  # An infinite range means no constraint, so f=0.
    
    # Inverse linear mapping: 0% -> f=1.0, 100 * 2.0 % -> f=0.0
    return 1.0 - (flying_range_percent / (100.0 * 2.0))

# ------------------------------------------------------
# Data preparation for flying-range model (edge_dim=3)
# ------------------------------------------------------
def prepare_data_flying_range(file_path: str,
                              pos_encoding_dim: int = 8,
                              split_ratios=(0.8, 0.2, 0.0),
                              num_entries: int | None = None,
                              device: torch.device | None = None,
                              k: int | None = None):
    """
    Builds PyG Data objects with:
      x:          PositionalEncoding(tour) -> [N, pos_encoding_dim]
      edge_index: either dense_to_sparse(C_t) or KNN over C_t
      edge_attr:  [E,3] with columns [truck=C_t_ij, drone=C_d_ij, f] (f is per-graph constant)
      y:          scalar label (uses 'label' if present else 'cost')

    Expects HDF5 groups containing at least:
      - 'dist_mtx' (C_t), 'tour', 'alpha'
      - either 'label' or 'cost'
      - 'alpha' and 'flying_range' to form f
    """
    assert abs(sum(split_ratios) - 1.0) < 1e-6, "split_ratios must sum to 1."

    if h5py is None:
        raise ImportError("h5py is required for prepare_data_flying_range() but is not installed")

    with h5py.File(file_path, 'r') as f:
        pos_enc = PositionalEncoding(pos_encoding_dim).to(device) if device else PositionalEncoding(pos_encoding_dim)

        # collect sample group keys only
        keys = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
        if num_entries is not None:
            keys = keys[:num_entries]

        data_list = []
        for key in keys:
            g = f[key]

            # Read tensors
            if device:
                C_t   = torch.tensor(g['dist_mtx'][:], dtype=torch.float32, device=device)
                tour  = torch.tensor(g['tour'][:],     dtype=torch.float32, device=device)
                alpha = torch.tensor(g['alpha'][()],   dtype=torch.float32, device=device)
                flying_range = torch.tensor(g['flying_range'][()], dtype=torch.float32, device=device)
                scaling_factor = torch.tensor(g['scaling_factor'][()], dtype=torch.float32, device=device)
            else:
                C_t   = torch.tensor(g['dist_mtx'][:], dtype=torch.float32)
                tour  = torch.tensor(g['tour'][:],     dtype=torch.float32)
                alpha = torch.tensor(g['alpha'][()],   dtype=torch.float32)
                flying_range = torch.tensor(g['flying_range'][()], dtype=torch.float32)
                scaling_factor = torch.tensor(g['scaling_factor'][()], dtype=torch.float32)

            # read label (prefer stored 'label', else 'cost')
            if 'label' in g:
                label_val = float(g['label'][()])
            elif 'cost' in g:
                label_val = float(g['cost'][()])
            else:
                raise KeyError(f"Group {key} missing both 'label' and 'cost'.")

            label = torch.tensor(label_val, dtype=torch.float32, device=C_t.device)

            # Node features from tour positions (fixed dim=pos_encoding_dim)
            x = pos_enc(tour)  # [N, pos_encoding_dim]
            if x.size(1) != pos_encoding_dim:
                raise RuntimeError(f"PositionalEncoding produced dim {x.size(1)} != {pos_encoding_dim}")

            # Drone distances C_d from alpha
            C_d = C_t / alpha

            # Edges & attributes
            if k is None:
                edge_index, edge_attr_t = dense_to_sparse(C_t)
                _,          edge_attr_d = dense_to_sparse(C_d)
                edge_attr_t = edge_attr_t.unsqueeze(1)  # [E,1]
                edge_attr_d = edge_attr_d.unsqueeze(1)  # [E,1]
            else:
                k_eff = min(k, C_t.size(0) - 1)
                edge_index = knn_graph_from_dist_matrix(C_t, k_eff)
                edge_attr_t = C_t[edge_index[0], edge_index[1]].unsqueeze(1)
                edge_attr_d = C_d[edge_index[0], edge_index[1]].unsqueeze(1)

            edge_attr = torch.cat([edge_attr_t, edge_attr_d], dim=1)  # [E,2]

            # Compute f scalar and append as 3rd column (constant per edge)
            f_val = compute_f_from_percentage(flying_range)
            f_col = torch.full((edge_attr.size(0), 1), f_val, dtype=edge_attr.dtype, device=edge_attr.device)
            edge_attr = torch.cat([edge_attr, f_col], dim=1)  # [E,3]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label)
            data_list.append(data)

    # Random split
    total_size = len(data_list)
    indices = torch.randperm(total_size)
    n_tr = int(total_size * split_ratios[0])
    n_v  = int(total_size * (split_ratios[0] + split_ratios[1]))
    train_idx = indices[:n_tr]
    val_idx   = indices[n_tr:n_v]
    test_idx  = indices[n_v:]

    train_data = CustomGraphDataset([data_list[i] for i in train_idx])
    val_data   = CustomGraphDataset([data_list[i] for i in val_idx])
    test_data  = CustomGraphDataset([data_list[i] for i in test_idx])

    return train_data, val_data, test_data
