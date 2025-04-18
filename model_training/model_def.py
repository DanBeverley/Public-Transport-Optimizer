"""
Defines ML model architectures using PyTorch.

Includes:
- Optimized LSTM model with Layer Normalization.
- GNN+LSTM model using GINEConv (handles edge features) and Layer Normalization.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, Dict, Data
try:
    import torch_geometric as pyg
    from torch_geometric.nn import GINEConv, global_add_pool, LayerNorm, Sequential
    from torch_geometric.nn.models import MLP
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False

from . import config
logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """Optimized LSTM model with optional Layer Normalization."""
    def __init__(self, input_dim:int, hidden_dim:int, num_layers:int, output_dim:int,
                 dropout_rate:float, use_layer_norm:bool=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        # Input: (batch_size, seq_len, input_dim)
        self.lstm = nn.LSTM(input_size = input_dim,
                            hidden_size = hidden_dim,
                            num_layers = num_layers,
                            batch_first = True,
                            dropout = dropout_rate if num_layers > 1 else 0)
        # Optional Layer Normalization on LSTM output (before final FC)
        if self.use_layer_norm:
            # Normalize over the hidden_dim features
            # LSTM output is (batch, seq, hidden), apply norm over last dim
            # Applying norm only on the *last* step's output before FC
            self.layer_norm = nn.LayerNorm(hidden_dim)
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        logger.info(f"LSTMModel initialized: input_dim={input_dim}, hidden_dim={hidden_dim},"
                    f"num_layers={num_layers}, output_dim={output_dim}, dropout={dropout_rate},"
                    f"use_layer_norm={use_layer_norm}")
    def forward(self, x:torch.Tensor, h_0:Optional[torch.Tensor]=None,
                c_0:Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Forward pass through the LSTM.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).
            h_0: Optional initial hidden state (num_layers, batch_size, hidden_dim).
            c_0: Optional initial cell state (num_layers, batch_size, hidden_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        # Initialize hidden/cell state if not provided
        if h_0 is None or c_0 is None:
            device = x.device
            h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device = device)
            c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device = device)
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h_0, c_0))
        # Use the hidden state of the last time step from the last layer
        last_hidden_state = hn[-1] # SHape (batch_size, hidden_dim)
        # Apply optional Layer Normalization
        if self.use_layer_norm:
            last_hidden_state = self.layer_norm(last_hidden_state)
        # Final prediction
        out = self.fc(last_hidden_state)
        return out
    
# GNN Encoder using GINEConv
class GNNEncoder(nn.Module):
    """
    GNN Encoder using GINEConv layers suitable for node and edge features,
    with Layer Normalization and ReLU activation.
    """
    def __init__(self, node_feature_dim:int, edge_feature_dim:int, hidden_dim:int, num_layers:int,
                 use_layer_norm:bool=True):
        super().__init__()
        if not _PYG_AVAILABLE:
            raise ImportError("Pytorch Geometric (PyG) is required for GNNEncoder")
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layer_norm else None
        self.use_layer_norm = use_layer_norm
        in_channels = node_feature_dim
        for i in range(num_layers):
            # GINEConv requires an MLP to process edge features before aggregation
            # Define a simple MLP for edge feature transformation
            edge_mlp = MLP([edge_feature_dim, hidden_dim, hidden_dim], act="relu")
            # eps=0 means simple sum aggregation, train_eps=True learns the epsilon parameter
            conv = pyg_nn.GINEConv(nn=MLP([in_channels, hidden_dim, hidden_dim], act="relu"),
                                   edge_dim=hidden_dim, train_eps=True)
            self.layers.append(conv)
            if self.use_layer_norm:
                # Apply layer norm on node embeddings *after* convolution + activation
                self.norms.append(LayerNorm(hidden_dim))
                in_channels = hidden_dim
            logger.info(f"GNNEncoder initialized: node_feat={node_feature_dim}, edge_feat={edge_feature_dim}"
                        f"hidden = {hidden_dim}, layers = {num_layers}, use_layer_norm={use_layer_norm} (using GINEConv)")
    def forward(self, x:torch.Tensor, edge_index:torch.Tensor, edge_attr:torch.Tensor)->torch.Tensor:
        """
        Forward pass through GINEConv layers.

        Args:
            x: Node features (num_nodes, node_feature_dim).
            edge_index: Edge connectivity (2, num_edges).
            edge_attr: Edge features (num_edges, edge_feature_dim).

        Returns:
            Node embeddings (num_nodes, hidden_dim).
        """
        if edge_attr is None:
            raise ValueError("GINEConv requires edge attributes (edge_attr)")
        for i, layer in enumerate(self.layers):
            # GINEConv expects edge_attr as input
            x = layer(x, edge_index, edge_attr = edge_attr)
            if self.use_layer_norm:
                x = self.norms[i](x)
            x = F.relu(x)
        return x 

# GNN + LSTM
class GNNLSTMModel(nn.Module):
    """
    Combines a GNN encoder (GINEConv) for static graph features with an LSTM
    for time series features. Concatenates GNN embedding with dynamic features.
    """
    def __init__(self, node_feature_dim:int, edge_feature_dim:int,
                 gnn_hidden_dim:int, gnn_layers:int, lstm_input_feature_dim:int,
                 lstm_hidden_dim:int, lstm_layers:int, output_dim:int, dropout_rate:float,
                 gnn_use_layer_norm:bool=True, lstm_use_layer_norm:bool = True):
        super().__init__()
        if not _PYG_AVAILABLE:
            raise ImportError("Pytorch Geometric (PyG) is required for GNNLSTMModel")
        self.gnn_encoder = GNNEncoder(node_feature_dim, edge_feature_dim, gnn_hidden_dim,
                                      gnn_layers, gnn_use_layer_norm)
        # Input dim for LSTM = GNN embedding dim + dynamic sequence feature dim
        combined_lstm_input_dim = gnn_hidden_dim + lstm_input_feature_dim
        self.lstm = nn.LSTM(input_size = combined_lstm_input_dim,
                            hidden_size=lstm_hidden_dim, num_layers = lstm_layers,
                            batch_first = True, dropout=dropout_rate if lstm_layers > 1 else 0)
        self.lstm_use_layer_norm = lstm_use_layer_norm
        if self.lstm_use_layer_norm:
            self.lstm_use_layer_norm = lstm_use_layer_norm
            if self.lstm_use_layer_norm:
                self.lstm_layer_norm = nn.LayerNorm(lstm_hidden_dim)
                self.fc = nn.Linear(lstm_hidden_dim, output_dim)
                self.gnn_hidden_dim = gnn_hidden_dim
                self.lstm_hidden_dim = lstm_hidden_dim
                self.lstm_layers = lstm_layers
                logger.info(f"GNNLSTMModel initialized: GNN hidden={gnn_hidden_dim}(LN={gnn_use_layer_norm}), "
                    f"LSTM input(combined)={combined_lstm_input_dim}, LSTM hidden={lstm_hidden_dim}(LN={lstm_use_layer_norm}), "
                    f"LSTM layers={lstm_layers}, Output={output_dim}, Dropout={dropout_rate}")
                
    def forward(self, x_seq:torch.Tensor, graph_data:Data, node_indices:torch.tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x_seq: Time series features (batch_size, seq_len, lstm_input_feature_dim). Dynamic features.
            graph_data: PyG Data object containing static graph (x, edge_index, edge_attr).
            node_indices: Tensor mapping each sequence in the batch to its corresponding node index
                          in the graph (batch_size,).

        Returns:
            Output prediction (batch_size, output_dim).
        """
        #1. Encode static graph features using GNN
        graph_data = graph_data.to(x_seq.device) # Move graph data to appropriate device
        if graph_data.x is None or graph_data.edge_attr is None:
            raise ValueError("Graph data must have node features (x) and edge features (edge_attr) for GNN Encoder")
        node_embeddings = self.gnn_encoder(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        # node_embeddings shape: (num_nodes, gnn_hidden_dim)

        # 2. Select relevant node embeddings for the batch sequences
        # node_indices might need adjustment if batching graphs (e.g., using graph_data.batch)
        selected_node_embeddings = node_embeddings[node_indices] # Shape: (batch_size, gnn_hidden_dim)

        # 3. Combine GNN embeddings with sequence features
        seq_len = x_seq.size(1)
        expanded_node_embeddings = selected_node_embeddings.unsqueeze(1).expand(-1, seq_len, -1)
        combined_features = torch.cat([expanded_node_embeddings, x_seq], dim=2)
        # combined_features shape: (batch_size, seq_len, gnn_hidden_dim + lstm_input_feature_dim)

        # 4. Pass combined features through LSTM
        # Initialize hidden/cell state
        h_0 = torch.zeros(self.lstm_layers, combined_features.size(0), self.lstm_hidden_dim, device=x_seq.device)
        c_0 = torch.zeros(self.lstm_layers, combined_features.size(0), self.lstm_hidden_dim, device=x_seq.device)
        lstm_out, (hn, cn) = self.lstm(combined_features, (h_0, c_0))

        # 5. Get output from last hidden state
        last_hidden_state = hn[-1] # Shape: (batch_size, lstm_hidden_dim)
        if self.lstm_use_layer_norm:
            last_hidden_state = self.lstm_layer_norm(last_hidden_state)
        out = self.fc(last_hidden_state) # Shape: (batch_size, output_dim)

        return out
        
        
