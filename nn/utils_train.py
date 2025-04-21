
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GraphNorm, LayerNorm, InstanceNorm
from torch_geometric.utils import scatter
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, Set2Set

# import h5py # For reading HDF5 files
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import PositionalEncoding

def get_normalization(normalization, num_features):
    if normalization == 'batch_norm':
        return nn.BatchNorm1d(num_features)
    elif normalization == 'layer_norm':
        return LayerNorm(num_features)
    elif normalization == 'instance_norm':
        return InstanceNorm(num_features)
    elif normalization == 'graph_norm':
        return GraphNorm(num_features)
    else:
        return nn.Identity()
    
def get_readout(readout_type, x, batch):
    if readout_type == 'mean':
        h_G = global_mean_pool(x, batch)
    elif readout_type == 'max':
        h_G = global_max_pool(x, batch)
    elif readout_type == 'sum':
        h_G = global_add_pool(x, batch)
    elif readout_type == 'attention':
        # Attention-based pooling
        node_weights = F.softmax(x, dim=1)
        h_G = scatter(node_weights * x, batch, dim=0, reduce='sum')
    #BUG: Not working
    elif readout_type == 'set2set': 
        set2set = Set2Set(x.size(1), processing_steps=2).to(x.device)
        h_G = set2set(x, batch)
    else:
        raise ValueError(f'Unknown readout type: {readout_type}')
    
    return h_G

def get_loss_function(loss_function):
    if loss_function == 'mse':
        return nn.MSELoss()
    elif loss_function == 'mae':
        return nn.L1Loss()
    elif loss_function == 'huber':
        return nn.HuberLoss()
    elif loss_function == 'smooth_l1':
        return nn.SmoothL1Loss()

def inspect_data(file_path, num_entries=None):
    with h5py.File(file_path, 'r') as f:
        if num_entries is not None:
            keys = list(f.keys())
            random.shuffle(keys)  # Shuffle the keys to randomize the order of entries
            keys = keys[:num_entries]  # Limit the number of entries
        else:
            keys = list(f.keys())

        for key in keys:
            group = f[key]
            dist_mtx = torch.tensor(group['dist_mtx'][:], dtype=torch.float32)
            tour = torch.tensor(group['tour'][:], dtype=torch.float32)
            label = torch.tensor(group['label'][()], dtype=torch.float32)
            # print(f"dist_mtx shape: {dist_mtx.shape}")
            # print(f"tour shape: {tour.shape}")
            # print(f"label shape: {label.shape}")
            # print(f"dist_mtx = {dist_mtx}")
            print(f"tour = {tour}")
            print(f"label = {label}")

# Function to load test data from HDF5 file
def load_data(file_path, pos_encoding_dim, device, num_entries=None):
    with h5py.File(file_path, 'r') as f:
        data_list = []

        pos_enc = PositionalEncoding(pos_encoding_dim).to(device)

        if num_entries is not None:
            keys = list(f.keys())
            random.shuffle(keys)  # Shuffle the keys to randomize the order of entries
            keys = keys[:num_entries]  # Limit the number of entries
        else:
            keys = list(f.keys())


        for key in keys:
            group = f[key]
            dist_mtx = torch.tensor(group['dist_mtx'][:], dtype=torch.float32).to(device) / 100
            tour = torch.tensor(group['tour'][:], dtype=torch.float32).to(device)
            pe = pos_enc(tour)
            label = torch.tensor(group['label'][()], dtype=torch.float32).to(device) / 100

            edge_index, edge_attr = dense_to_sparse(dist_mtx)
            data = Data(x=pe, edge_index=edge_index, edge_attr=edge_attr, y=label)
            data_list.append(data)
    
    return data_list

class CustomGraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    

def knn_graph_from_dist_matrix(dist_matrix, k):
    num_nodes = dist_matrix.size(0)
    edge_index = []
    for i in range(num_nodes):
        distances = dist_matrix[i].cpu().numpy()
        nearest_indices = np.argsort(distances)[1:k+1]  # Exclude the node itself
        for j in nearest_indices:
            edge_index.append([i, j])
            edge_index.append([j, i])  # Add both directions for undirected graph
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

def prepare_data(file_path, pos_encoding_dim, split_ratios, num_entries=None, device=None, k=None):
    with h5py.File(file_path, 'r') as f:
        data_list = []

        if device:
            pos_enc = PositionalEncoding(pos_encoding_dim).to(device)
        else: 
            pos_enc = PositionalEncoding(pos_encoding_dim)

        keys = list(f.keys())
        # random.shuffle(keys)  # Shuffle the keys to randomize the order of entries
        if num_entries is not None:
            keys = keys[:num_entries]  # Limit the number of entries

        # flag = True

        for key in keys:
            group = f[key]
            
            if device:
                C_t = torch.tensor(group['dist_mtx'][:], dtype=torch.float32).to(device)
                tour = torch.tensor(group['tour'][:], dtype=torch.float32).to(device)
                pe = pos_enc(tour)
                label = torch.tensor(group['label'][()], dtype=torch.float32).to(device)
                alpha = torch.tensor(group['alpha'][()], dtype=torch.float32).to(device)
            else:
                C_t = torch.tensor(group['dist_mtx'][:], dtype=torch.float32)
                tour = torch.tensor(group['tour'][:], dtype=torch.float32)
                pe = pos_enc(tour)
                label = torch.tensor(group['label'][()], dtype=torch.float32)
                alpha = torch.tensor(group['alpha'][()], dtype=torch.float32)

            C_d = C_t / alpha
            if k is None:
                edge_index, edge_attr_t = dense_to_sparse(C_t)
                _, edge_attr_d = dense_to_sparse(C_d)
                edge_attr_t, edge_attr_d = edge_attr_t.unsqueeze(1), edge_attr_d.unsqueeze(1)
            else:
                k = min(k, C_t.size(0) - 1)
                edge_index = knn_graph_from_dist_matrix(C_t, k)
                edge_attr_t = C_t[edge_index[0], edge_index[1]].unsqueeze(1)
                edge_attr_d = C_d[edge_index[0], edge_index[1]].unsqueeze(1)

            edge_attr = torch.cat((edge_attr_t, edge_attr_d), dim=1)
            data = Data(x=pe, edge_index=edge_index, edge_attr=edge_attr, y=label)
            data_list.append(data)
    
    total_size = len(data_list)
    indices = torch.randperm(total_size)
    train_idx = indices[:int(total_size * split_ratios[0])]
    val_idx = indices[int(total_size * split_ratios[0]):int(total_size * (split_ratios[0] + split_ratios[1]))]
    test_idx = indices[int(total_size * (split_ratios[0] + split_ratios[1])):]

    train_data = CustomGraphDataset([data_list[idx] for idx in train_idx])
    val_data = CustomGraphDataset([data_list[idx] for idx in val_idx])
    test_data = CustomGraphDataset([data_list[idx] for idx in test_idx])

    return train_data, val_data, test_data

    # train_data = CustomGraphDataset(data_list)
    # return train_data