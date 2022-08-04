import torch
import torch.nn as nn
from math import log
from copy import deepcopy
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x += self.pe[:, : x.size(1)]
        return self.dropout(x)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------#

class ResidualFFNN(nn.Module):
    def __init__(self, d_model=512, d_fc=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_fc)
        self.fc2 = nn.Linear(d_fc, d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.act = {
            "relu": F.relu,
            "gelu": F.gelu
        }[activation]
        
    def forward(self, x):
        x += self.dropout2(self.fc2(self.dropout1(self.act(self.fc1(x)))))
        return self.layer_norm(x)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------#

class ResidualMHA(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model,
                                         num_heads=n_heads,
                                         dropout=dropout,
                                         batch_first=True)
        
    def forward(self, x, mask=None):
        x += self.mha(query=x, 
                      key=x, 
                      value=x, 
                      need_weights=False,
                      key_padding_mask=mask)[0]
        
        return self.layer_norm(x)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------#

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1, d_fc=2048, activation="relu"):
        super().__init__()
        self.res_mha = ResidualMHA(d_model=d_model,
                                   n_heads=n_heads,
                                   dropout=dropout)
        
        self.res_ffnn = ResidualFFNN(d_model=d_model,
                                     d_fc=d_fc,
                                     dropout=dropout,
                                     activation=activation)
        
    def forward(self, x, mask=None):
        return self.res_ffnn(self.res_mha(x, mask=mask))

#--------------------------------------------------------------------------------------------------------------------------------------------------------------#

class EncoderBlock(nn.Module):
    def __init__(self, encoder_layer, n_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(encoder_layer) for _ in range(n_layers)])
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

#--------------------------------------------------------------------------------------------------------------------------------------------------------------#