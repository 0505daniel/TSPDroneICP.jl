# infer_flying_range.py
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import PositionalEncoding

# reuse the exact normalization you trained with
from utils_train_fr import compute_f_from_percentage

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from keys in a DataParallel checkpoint."""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k[7:] if k.startswith('module.') else k] = v
    return new_state_dict


def get_graph(initial_route, C_t, C_d, pos_encoding_dim, scale_factor, flying_range):
    """
    Build a PyG graph with 3 edge features: [truck, drone, f].
    Distances in C_t/C_d must be raw; we divide by scale_factor here
    (same convention as training). 'f' is constant per-edge within the graph.
    """
    # node features = positional encoding of the route indices
    x = torch.tensor(initial_route, dtype=torch.float32).unsqueeze(1)
    pe = PositionalEncoding(pos_encoding_dim)(x)

    # normalize costs
    Ct = torch.tensor(C_t, dtype=torch.float32)
    max_dist = float(torch.max(Ct))
    Ct = Ct / float(scale_factor)
    Cd = torch.tensor(C_d, dtype=torch.float32) / float(scale_factor)

    edge_index, edge_attr_t = dense_to_sparse(Ct)
    _,          edge_attr_d = dense_to_sparse(Cd)
    edge_attr = torch.cat((edge_attr_t.unsqueeze(1), edge_attr_d.unsqueeze(1)), dim=1)  # [E,2]

    flying_range_percentage = float(flying_range) * 2.0 / max_dist * 100

    # third edge feature = f (same for all edges in this graph)
    f_val = compute_f_from_percentage(flying_range_percentage)
    f_col = torch.full((edge_attr.size(0), 1), float(f_val), dtype=edge_attr.dtype)
    edge_attr = torch.cat([edge_attr, f_col], dim=1)  # [E,3]

    data = Data(x=pe, edge_index=edge_index, edge_attr=edge_attr)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


@torch.no_grad()
def predict_chainlet_length(model, device, initial_route, C_t, C_d,
                            pos_encoding_dim, scale_factor, flying_range):
    g = get_graph(initial_route, C_t, C_d, pos_encoding_dim, scale_factor, flying_range)
    g = g.to(device)
    out = model(g)
    return out.item()


@torch.no_grad()
def batch_prediction(device, model, pos_encoding_dim, chainlets, scale_factors, batch_size=23):
    """
    chainlets: list of dicts with keys:
      - 'initial_route'  (list[int])
      - 'C_t'            (2D list/array)
      - 'C_d'            (2D list/array)
      - 'flying_range'   (float)
    scale_factors: list[float] (same order as chainlets)
    Returns: predictions scaled back to raw units.
    """
    data_list = [
        get_graph(ch['initial_route'], ch['C_t'], ch['C_d'],
                  pos_encoding_dim, scale_factors[i], ch['flying_range'])
        for i, ch in enumerate(chainlets)
    ]

    bs = min(batch_size, len(data_list))
    loader = DataLoader(data_list, batch_size=bs, shuffle=False)

    preds = []
    for data in loader:
        data = data.to(device)
        preds.append(model(data).cpu())

    preds = torch.cat(preds, dim=0).flatten()                 # normalized outputs
    scaled = preds * torch.tensor(scale_factors, dtype=torch.float32)  # de-normalize
    return scaled
