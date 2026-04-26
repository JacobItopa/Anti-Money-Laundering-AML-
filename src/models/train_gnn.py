import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score

class EdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim):
        super(EdgeClassifier, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        # MLP for edges: source node embedding + target node embedding + edge features
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2 + edge_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        # Generate node embeddings
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        
        # Get source and target embeddings for each edge
        src, dst = edge_index
        src_emb = x[src]
        dst_emb = x[dst]
        
        # Concatenate src, dst, and edge features
        edge_input = torch.cat([src_emb, dst_emb, edge_attr], dim=-1)
        return self.mlp(edge_input).squeeze()

def build_graph(df: pd.DataFrame, edge_features_cols: list) -> Data:
    print("Building PyG Data object...")
    
    # Map account IDs to integers
    le = LabelEncoder()
    # Combine Account and Account.1 to fit a single encoder
    all_accounts = pd.concat([df['Account'], df['Account.1']])
    le.fit(all_accounts)
    
    src = le.transform(df['Account'])
    dst = le.transform(df['Account.1'])
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
    
    # Node features (dummy ones since we don't have distinct node features)
    num_nodes = len(le.classes_)
    x = torch.ones((num_nodes, 16), dtype=torch.float) # 16-dim dummy feature
    
    # Edge features
    edge_attr = torch.tensor(df[edge_features_cols].astype(float).values, dtype=torch.float)
    
    # Labels
    y = torch.tensor(df['Is Laundering'].values, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data

def train_gnn(data, epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}...")
    model = EdgeClassifier(in_channels=16, hidden_channels=32, edge_dim=data.edge_attr.size(1)).to(device)
    data = data.to(device)
    
    # Calculate scale_pos_weight
    pos_weight = (data.y == 0).sum() / (data.y == 1).sum()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train test split (80/20) for edges
    num_edges = data.edge_index.size(1)
    indices = np.random.permutation(num_edges)
    train_idx = indices[:int(0.8 * num_edges)]
    test_idx = indices[int(0.8 * num_edges):]
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # Forward pass on all nodes/edges
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            out_test = out[test_idx]
            probs = torch.sigmoid(out_test).cpu().numpy()
            y_true = data.y[test_idx].cpu().numpy()
            pr_auc = average_precision_score(y_true, probs)
            
        print(f"Epoch {epoch+1:02d}, Loss: {loss.item():.4f}, Test PR-AUC: {pr_auc:.4f}")
        
    return model, train_idx, test_idx
