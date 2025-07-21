# models.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

# --- Standard Models (kept for comparative analysis) ---
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def get_model(name, in_channels, hidden_channels, out_channels):
    if name == "GCN":
        return GCN(in_channels, hidden_channels, out_channels)
    elif name == "GAT":
        return GAT(in_channels, hidden_channels, out_channels)
    elif name == "GraphSAGE":
        return GraphSAGE(in_channels, hidden_channels, out_channels)
    else:
        raise ValueError(f"Unknown model: {name}")

# --- New Dynamic Model ---
class DynamicGNN(torch.nn.Module):
    """
    A GNN model that can be built dynamically with a variable number of layers.
    """
    def __init__(self, in_channels, out_channels, layer_configs):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        
        current_channels = in_channels
        
        for i, config in enumerate(layer_configs):
            layer_type = config['type']
            hidden_channels = config['hidden_channels']
            
            if layer_type == "GCN":
                self.layers.append(GCNConv(current_channels, hidden_channels))
            elif layer_type == "GAT":
                # For simplicity in dynamic building, we use concat=False for intermediate layers
                self.layers.append(GATConv(current_channels, hidden_channels, heads=1, concat=False))
            elif layer_type == "GraphSAGE":
                self.layers.append(SAGEConv(current_channels, hidden_channels))
            
            current_channels = hidden_channels
            
        # Final layer to map to output classes
        self.out_layer = GCNConv(current_channels, out_channels)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index).relu()
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.out_layer(x, edge_index)
        return x
