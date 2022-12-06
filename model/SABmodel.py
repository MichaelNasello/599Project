import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class MAB(nn.Module):
    def __init__(self, embed_dim):
        super(MAB, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, 8)
        self.rff = nn.Linear(embed_dim, embed_dim)
        self.lnorm = nn.LayerNorm(embed_dim)
        
    def forward(self, Y, X):
        attn_output, _ = self.multihead_attn(Y, X, X)
        H = attn_output + Y
        H_output = self.rff(H)
        output = self.lnorm(H_output + H)
        return output 
    
    
class SAB(nn.Module):
    def __init__(self, embed_dim):
        super(SAB, self).__init__()
        self.mab = MAB(embed_dim)
        
    def forward(self, X):
        output = self.mab(X, X)
        return output

    
class TransformerClassifier(nn.Module):
    def __init__(self, d_model, n_layers, emb_dims, hid_dims, dropout = 0.3, lr = 1e-5):

        super().__init__()
        self.d_model = d_model
        
        self.pos_embed = PositionalEncoding(emb_dims, dropout)
        sabs = []
        for i in range(n_layers):
            sabs.append(SAB(emb_dims))
        self.sab_layers = nn.Sequential(*sabs)
        self.linear0 = nn.Linear(d_model, emb_dims)
        self.linear1 = nn.Linear(emb_dims, hid_dims)
        self.linear2 = nn.Linear(hid_dims, 1)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.LayerNorm(emb_dims)
        self.norm2 = nn.LayerNorm(hid_dims)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        dims = x.shape
        x = x.to('cuda')
        x = self.linear0(x)
        x = self.pos_embed(x)
        x = self.sab_layers(x)
        
        x = self.norm1(x)
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.relu1(x)
        
        x = self.norm2(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        return torch.sigmoid(x).view(dims[0], dims[1])