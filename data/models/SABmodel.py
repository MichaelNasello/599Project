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
        # Create a long enough P
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
        self.multihead_attn = nn.MultiheadAttention(embed_dim, 16)
        # Probably could try different ff.
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
    
class ISAB(nn.Module):
    def __init__(self, embed_dim, induce_pts):
        super(ISAB, self).__init__()
        self.mab1 = MAB(embed_dim)
        self.mab2 = MAB(embed_dim)
        self.i_param = nn.Parameter(torch.Tensor(induce_pts, 1, embed_dim))
        nn.init.xavier_uniform_(self.i_param)
        
    def forward(self, X):
        H = self.mab1(self.i_param.repeat(1, X.shape[1], 1), X)
        output = self.mab2(X, H)
        return output
    
class TransformerClassifier(nn.Module):
    def __init__(self, d_model: int, emb_dims: int, hid_dims: int, dropout: float = 0.3):
        """Initialize a TransformerEncoderLayer.

        Parameters
        ----------
        d_model : int
            The number of expected features in the input.
        n_head : int
            The number of heads in the multiheadattention models.
        dim_feedforward : int, optional
            The dimension of the feedforward network (default=2048).
        dropout : float, optional
            The dropout value (default=0.1).

        """
        super().__init__()
        self.d_model = d_model
        
        self.pos_embed = PositionalEncoding(emb_dims, dropout)
        self.sab1 = SAB(emb_dims)
        self.sab2 = SAB(emb_dims)
        self.sab3 = SAB(emb_dims)
        self.linear0 = nn.Linear(d_model, emb_dims)
        self.linear1 = nn.Linear(emb_dims, hid_dims)
        self.linear2 = nn.Linear(hid_dims, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.norm1 = nn.LayerNorm(emb_dims)
        self.norm2 = nn.LayerNorm(hid_dims)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout) 
        
        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        dims = x.shape
        x = x.to('cuda')
#         x = torch.cat([torch.ones(x.shape[0], 1, 17).to('cuda'), x], axis=1)
        x = self.linear0(x)
        x = self.pos_embed(x)
        x = self.sab1(x)
        x = self.sab2(x)
        x = self.sab3(x)
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        x = self.linear1(x)
        
        x = self.norm2(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        return torch.sigmoid(x).view(dims[0], dims[1])