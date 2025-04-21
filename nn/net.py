import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from utils_train import get_normalization, get_readout
    
class TSPDGraphTransformerNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, beta=True, dropout=0.2, normalization='batch_norm', num_gat_layers=2, activation='elu', readout_type='attention'):
        super(TSPDGraphTransformerNetwork, self).__init__()

        self.activation = getattr(F, activation)
        self.transformer_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.readout_type = readout_type

        # Input layer
        self.transformer_layers.append(TransformerConv(in_channels, hidden_channels // heads, heads=heads, concat=True, beta=beta, dropout=dropout, edge_dim=2))
        self.norm_layers.append(get_normalization(normalization, hidden_channels))

        # Hidden layers
        for _ in range(num_gat_layers - 1):
            self.transformer_layers.append(TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, concat=True, beta=beta, dropout=dropout, edge_dim=2))
            self.norm_layers.append(get_normalization(normalization, hidden_channels))

        # Output layer
        self.transformer_layers.append(TransformerConv(hidden_channels, out_channels // heads, heads=heads, concat=True, beta=beta, dropout=dropout, edge_dim=2))
        self.norm_layers.append(get_normalization(normalization, out_channels))

        self.output_layer = nn.Linear(out_channels, 1)  # Output layer for scalar projection
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # If batch is None, create a default batch where all nodes belong to the same graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # # Debug: Print shapes of tensors
        # print(f'x shape: {x.shape}')
        # print(f'edge_index shape: {edge_index.shape}')
        # print(f'edge_attr shape: {edge_attr.shape}')
        # print(f'batch shape: {batch.shape}')

        for i, transformer_layer in enumerate(self.transformer_layers):
            x = transformer_layer(x, edge_index, edge_attr=edge_attr)
            # print(f'After layer {i}, x shape: {x.shape}')
            x = self.norm_layers[i](x)
            x = self.activation(x)

        # Graph readout
        h_G = get_readout(self.readout_type, x, batch)

        # Apply the linear transformation
        output = self.output_layer(h_G) + self.bias

        return output
    
    