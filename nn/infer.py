import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import PositionalEncoding

def remove_module_prefix(state_dict):
    """
    Remove 'module.' prefix from keys in state_dict.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove the 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict

def get_graph(initial_route, C_t, C_d, pos_encoding_dim, scale_factor):
    x = torch.tensor(initial_route, dtype=torch.float32).unsqueeze(1)  # Node features
    pos_enc = PositionalEncoding(pos_encoding_dim)
    pe = pos_enc(x)
    C_t = torch.tensor(C_t, dtype=torch.float32) / scale_factor
    C_d = torch.tensor(C_d, dtype=torch.float32) / scale_factor
    edge_index, edge_attr_t = dense_to_sparse(C_t)
    _, edge_attr_d = dense_to_sparse(C_d)
    edge_attr_t, edge_attr_d = edge_attr_t.unsqueeze(1), edge_attr_d.unsqueeze(1)
    edge_attr = torch.cat((edge_attr_t, edge_attr_d), dim=1)
    data = Data(x=pe, edge_index=edge_index, edge_attr=edge_attr)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def predict_chainlet_length(model, device, initial_route, C_t, C_d, pos_encoding_dim, scale_factor):
    graph = get_graph(initial_route, C_t, C_d, pos_encoding_dim, scale_factor)
    graph.to(device)
    with torch.no_grad():
        output = model(graph)
    return output.item()   

def batch_prediction(device, model, pos_encoding_dim, chainlets, scale_factors, batch_size=23):
    predictions = []
    data_list = [get_graph(chainlet['initial_route'], chainlet['C_t'], chainlet['C_d'], pos_encoding_dim, scale_factors[i]) for i, chainlet in enumerate(chainlets)]
    # print(len(data_list))

    if len(data_list) < batch_size:
        batch_size = len(data_list)
        
    # print(batch_size)
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            predictions.append(output.cpu())
    
    predictions = torch.cat(predictions).flatten()

    # Scale up the predictions
    scaled_predictions = predictions * torch.tensor(scale_factors, dtype=torch.float32) 

    return scaled_predictions