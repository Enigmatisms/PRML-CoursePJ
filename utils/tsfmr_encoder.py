#-*-coding:utf-8-*-
"""
    This implementation is originated from the reproduction of mine:
    Github Maevit: https://github.com/Enigmatisms/Maevit
    I am just reusing the Transformer Layer
    Escaping the Big Data Paradigm with Compact Transformers
    @author Enigmatisms
"""

import torch
from torch import nn
from math import sqrt
from torch.nn import functional as F
from timm.models.layers import DropPath

# 2 in here is the expansion ratio
def makeMLP(in_chan, mlp_dropout):
    return nn.Sequential(
        nn.Linear(in_chan, 2 * in_chan),
        nn.GELU(),
        nn.Dropout(mlp_dropout),
        nn.Linear(2 * in_chan, in_chan),
        nn.Dropout(mlp_dropout)
    )

"""
    Transformer Encoder for ViT (Version 2.0)
    Yeah, originally for ViT, now it is for SwinTransformer1D
"""
class TransformerEncoder(nn.Module):
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def __init__(self, emb_dim, mlp_chan, 
        mlp_dropout = 0.1, 
        att_drop = 0.1, 
        proj_drop = 0.1, 
        path_drop = 0.1, 
        head_num = 2
    ):
        super().__init__()
        assert(emb_dim % head_num == 0)
        self.emb_dim = emb_dim
        self.emb_dim_h_k = emb_dim // head_num
        self.normalize_coeff = sqrt(self.emb_dim_h_k)
        self.head_num = head_num
        self.qkv_attn = nn.Linear(emb_dim, emb_dim * 3, bias = False)
        self.proj_o = nn.Linear(emb_dim, emb_dim)
        self.pre_ln = nn.LayerNorm(emb_dim)
        self.post_ln = nn.LayerNorm(emb_dim)
        self.mlp = makeMLP(mlp_chan, mlp_dropout)
        self.drop_path = DropPath(path_drop)
        self.attn_drop = nn.Dropout(att_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self.init_weight)
        # Add attention dropout
    

    def attention(self, X:torch.Tensor):
        batch_size, seq_len, _ = X.shape
        # input (b, seq_len, embedding_dim) while output of qkv attn is (b, seq_len, 3 * embedding_dim)
        # output shape: (3, batch_size, head_num, seq_len, emb_dim / head_num)
        qkvs:torch.Tensor = self.qkv_attn(X).view(batch_size, seq_len, 3, self.head_num, self.emb_dim_h_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkvs[0], qkvs[1], qkvs[2]
        proba = F.softmax((q @ k.transpose(-1, -2)) / self.normalize_coeff, dim = -1)
        proba = self.attn_drop(proba)
        # proba: batch_size, head_num, seq_len, seq_len -> output (batch_size, head_num, seq_len, emb_dim/ head_num)
        proj_o = self.proj_o((proba @ v).transpose(1, 2).reshape(batch_size, seq_len, -1))
        return self.proj_drop(proj_o)
    
    def forward(self, X):
        tmp = self.pre_ln(X)
        tmp = self.attention(tmp)
        X = self.post_ln(X + self.drop_path(tmp))
        tmp2 = self.mlp(X)
        return X + self.drop_path(tmp2)
        