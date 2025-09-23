# net_fr.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch_geometric.nn import TransformerConv, GraphNorm
from torch_geometric.utils import scatter  # per-graph reductions from edge-wise values

# -----------------------------
# FiLM (graph-conditional)
# -----------------------------
class FiLMAdapter(nn.Module):
    """
    Post-norm, pre-activation FiLM (per-channel):
      gamma = 1 + s * g,  beta = t * g
    Exact identity when g(f)=0.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.s = nn.Parameter(torch.zeros(dim))  # [d]
        self.t = nn.Parameter(torch.zeros(dim))  # [d]

    def forward(self, z: torch.Tensor, g_graph: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        z:       [N, d]  node features after normalization
        g_graph: [B]     per-graph scalar gate g(f)
        batch:   [N]     node->graph ids in 0..B-1
        """
        g_nodes = g_graph.view(-1, 1)[batch]  # [N,1]
        gamma = 1 + g_nodes * self.s          # [N,d]
        beta  =     g_nodes * self.t          # [N,d]
        return z * gamma + beta


# -----------------------------
# Scalar φ-MLP  (optional)
# -----------------------------
class ScalarPhiMLP(nn.Module):
    """
    φ(f): scalar → scalar. Tiny MLP to learn a gain schedule over f∈[0,1].
      - layers: 0, 1, or 2 hidden layers  (0 → φ≡1)
      - hidden: width of hidden layers (e.g., 8/16/32)
      - activation: 'relu' | 'leaky_relu' | 'elu' (hidden only)
      - head: 'relu' (φ>=0) | 'linear' (unconstrained φ)
    """
    def __init__(self,
                 layers: int = 1,
                 hidden: int = 16,
                 activation: str = "relu",
                 leaky_slope: float = 0.01):
        super().__init__()
        self.layers = int(layers)
        self.hidden = int(hidden)
        act = activation.lower()
        self._hidden_act = {
            "relu": nn.ReLU(inplace=True),
            "leaky_relu": nn.LeakyReLU(leaky_slope, inplace=True),
            "leakyrelu": nn.LeakyReLU(leaky_slope, inplace=True),
            "elu": nn.ELU(alpha=1.0, inplace=True),
        }[act]

        if self.layers <= 0:
            self.mlp = None
        else:
            mods = [nn.Linear(1, self.hidden), self._hidden_act]
            if self.layers >= 2:
                mods += [nn.Linear(self.hidden, self.hidden), self._hidden_act]
            mods += [nn.Linear(self.hidden, 1)]
            self.mlp = nn.Sequential(*mods)
            # small positive bias at input to avoid dead ReLUs near f≈0
            with torch.no_grad():
                if isinstance(self.mlp[0], nn.Linear):
                    self.mlp[0].bias.fill_(0.1)

    def forward(self, f_graph: torch.Tensor) -> torch.Tensor:
        """
        f_graph: [B] in [0,1]
        return:  φ(f): [B]
        """
        if self.mlp is None:
            return torch.ones_like(f_graph)  # φ ≡ 1
        u = self.mlp(f_graph.view(-1, 1)).view(-1)  # [B]
        # return F.relu(u)      # φ >= 0 (can be 0 if u<0)
        return F.softplus(u)


# -----------------------------
# Graph Transformer (Flying-Range version)
# -----------------------------
class TSPDGraphTransformerNetworkFlyingRange(nn.Module):
    """
    Flying-range variant:
      - fixed normalization: GraphNorm
      - fixed activation:   ELU
      - fixed readout:      'attention'
      - edge features:      (truck, drone, flying_range)  → edge_dim must be 3
      - FiLM after norm, before activation (graph-conditional)
      - optional φ-MLP to form g(f)=f*φ(f) for FiLM
    """
    def __init__(self,
                 dropout: float = 0.0,
                 edge_dim: int = 3,
                 phi_layers: int = 0,
                 phi_hidden: int = 16,
                 phi_activation: str = 'relu',
                 phi_leaky_slope: float = 0.01, 
                 film_init: str = 'zeros',      # 'zeros' | 'normal'
                 film_std: float = 1e-3,
                 edge3_init: str = 'default'):    # 'default' | 'zero'):
        super().__init__()
        assert edge_dim == 3, "Flying-range net expects edge_dim=3 with f as the 3rd column."

        # FIXED CONFIGURATION FROM THE PREVIOUS MODEL #
        ##########################################################
        self.channels = 8
        self.num_gt_layers = 5
        self.heads = 4
        self.edge_dim = edge_dim
        self.act = F.elu  # fixed activation

        def attention_readout(x, batch):
            node_weights = F.softmax(x, dim=1)
            h_G = scatter(node_weights * x, batch, dim=0, reduce='sum')
            return h_G
        
        self.readout_type = attention_readout  # fixed readout
        #########################################################

        # Convolution + GraphNorm + FiLM stacks
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.films = nn.ModuleList()

        for i in range(self.num_gt_layers):
            self.convs.append(TransformerConv(self.channels,
                                              self.channels // self.heads,
                                              heads=self.heads,
                                              concat=True,
                                              beta=False,
                                              dropout=dropout,
                                              edge_dim=edge_dim))
            self.norms.append(GraphNorm(self.channels))
            self.films.append(FiLMAdapter(self.channels))

        # φ-MLP gate (scalar). If layers=0 → φ≡1 → g(f)=f.
        self.phi_gate = ScalarPhiMLP(layers=phi_layers,
                                     hidden=phi_hidden,
                                     activation=phi_activation,
                                     leaky_slope=phi_leaky_slope)

        # Output Layer
        self.out_proj = nn.Linear(self.channels, 1)
        self.bias = nn.Parameter(torch.zeros(1))

        # --- init options ---
        self._init_film(film_init, film_std)
        if edge3_init == 'zero':
            self._zero_third_edge_column()

    def _init_film(self, mode: str, std: float):
        for film in self.films:
            if isinstance(film, FiLMAdapter):
                if mode == 'zeros':
                    nn.init.zeros_(film.s); nn.init.zeros_(film.t)
                elif mode == 'normal':
                    nn.init.normal_(film.s, mean=0.0, std=std)
                    nn.init.normal_(film.t, mean=0.0, std=std)
                else:
                    raise ValueError(f"Unknown film_init: {mode}")

    @torch.no_grad()
    def _zero_third_edge_column(self):
        # Safer access: TransformerConv exposes lin_edge (Linear) when edge_dim is set
        for conv in self.convs:
            if hasattr(conv, "lin_edge") and isinstance(conv.lin_edge, nn.Linear):
                W = conv.lin_edge.weight    # [out, edge_dim]
                if W.size(1) >= 3:
                    W[:, 2].zero_()

    def forward(self, data):
        """
        data.x         [N, in_channels]
        data.edge_index
        data.edge_attr [E, 3]  (3rd column is f)
        data.batch     [N]     (optional; if missing, assumes single graph)
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        edge_f = edge_attr[:, 2] # [E]
        edge_batch = batch[edge_index[0]] # [E]
        num_graphs = int(edge_batch.max().item()) + 1
        flying_range = scatter(edge_f, edge_batch, dim=0, dim_size=num_graphs, reduce='mean') # [B]

        # gate g(f) = f * φ(f)
        g_f = flying_range * self.phi_gate(flying_range) # [B]

        # Stacked TransformerConv → GraphNorm → FiLM → ELU
        for conv, norm, film in zip(self.convs, self.norms, self.films):
            x = conv(x, edge_index, edge_attr=edge_attr)
            # print(f'After layer {i}, x shape: {x.shape}')
            z = norm(x, batch)                 # GraphNorm(x, batch)
            z = film(z, g_f, batch)        # graph-conditional modulation
            x = self.act(z)

        # Graph readout (fixed: 'attention') and projection
        h_G = self.readout_type(x, batch)  # [B, out_channels]
        out = self.out_proj(h_G) + self.bias            # [B, 1]
        return out


# ============================
# Legacy → Flying-Range helpers
# ============================

def _strip_module(sd):
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

@torch.no_grad()
def load_legacy_into_flying_range(model_new: TSPDGraphTransformerNetworkFlyingRange,
                                  legacy_ckpt_path: str,
                                  device: torch.device):
    """
    Copy legacy (edge_dim=2) weights into flying-range (edge_dim=3) model:
      convs.i.*  <- transformer_layers.i.*
      norms.i.*  <- norm_layers.i.*
      out_proj.* <- output_layer.*
      bias       <- bias
    For lin_edge.weight: copy first two columns; keep 3rd as-initialized.
    """
    sd = torch.load(legacy_ckpt_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = _strip_module(sd)

    new_sd = model_new.state_dict()
    mapped = {}
    for k_old, v_old in sd.items():
        if k_old.startswith("transformer_layers."):
            k_new = k_old.replace("transformer_layers.", "convs.")
            if k_new in new_sd:
                if k_new.endswith("lin_edge.weight") and v_old.dim()==2:
                    W_new = new_sd[k_new].clone()
                    # copy legacy 2 cols into the first two cols
                    W_new[:, :min(2, W_new.size(1))] = v_old[:, :min(2, W_new.size(1))]
                    mapped[k_new] = W_new
                else:
                    mapped[k_new] = v_old
        elif k_old.startswith("norm_layers."):
            k_new = k_old.replace("norm_layers.", "norms.")
            if k_new in new_sd:
                mapped[k_new] = v_old
        elif k_old.startswith("output_layer."):
            k_new = k_old.replace("output_layer.", "out_proj.")
            if k_new in new_sd:
                mapped[k_new] = v_old
        elif k_old == "bias" and "bias" in new_sd:
            mapped["bias"] = v_old

    model_new.load_state_dict(mapped, strict=False)

    missing, unexpected = model_new.load_state_dict(mapped, strict=False)
    print("missing:", missing)       # params in new model not found in legacy
    print("unexpected:", unexpected) # params in legacy not used by new model

def freeze_legacy_and_enable_new(model_new: TSPDGraphTransformerNetworkFlyingRange):
    """
    Freeze ALL params, then enable ONLY:
      - FiLM (s,t)
      - φ-MLP (if layers>0)
      - the 3rd column of each conv.lin_edge.weight
    """
    for p in model_new.parameters():
        p.requires_grad = False

    # FiLM
    for film in model_new.films:
        if isinstance(film, FiLMAdapter):
            film.s.requires_grad = True
            film.t.requires_grad = True

    # φ-MLP (if present)
    if hasattr(model_new, "phi_gate") and model_new.phi_gate is not None:
        mlp = getattr(model_new.phi_gate, "mlp", None)
        if mlp is not None:
            for p in mlp.parameters():
                p.requires_grad = True

    # Only 3rd edge column gets gradients
    def mask_hook_for_third_col(param):
        mask = torch.zeros_like(param)
        if param.size(1) >= 3:
            mask[:, 2] = 1.0
        def hook(grad): return grad * mask
        return hook

    for conv in model_new.convs:
        if hasattr(conv, "lin_edge") and isinstance(conv.lin_edge, nn.Linear):
            W = conv.lin_edge.weight
            W.requires_grad = True
            W.register_hook(mask_hook_for_third_col(W))
    # out_proj & its bias remain frozen by default