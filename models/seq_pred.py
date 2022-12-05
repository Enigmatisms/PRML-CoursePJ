#-*-coding:utf-8-*-
"""
    Modified from Swin Transformer (Transformer models failed to predict)
    @author: Qianyue He
    @date: 2022-11-19 - @modified 2022-12-4
"""

import torch
from torch import nn
from timm.models.layers.drop import DropPath
from torch.nn import functional as F

def makeMLP(in_chan, out_chan, mlp_dropout = 0.1, act = nn.GELU(), bias = True):
    layers = [nn.Linear(in_chan, out_chan, bias = bias)]
    if mlp_dropout > 1e-4:
        layers.append(nn.Dropout(mlp_dropout))
    if act is not None:
        layers.append(act)
    return layers
    
def makeConv1D(in_chan, out_chan, ksize = 3, act = nn.GELU(), norm = None, max_pool = 0, padding = -1):
    layer = [nn.Conv1d(in_chan, out_chan, kernel_size = ksize, padding = ksize >> 1 if padding < 0 else padding)]
    if norm is not None:
        layer.append(norm)
    if max_pool > 0:
        layer.append(nn.MaxPool1d(max_pool))
    if act is not None:
        layer.append(act)
    return layer

class PatchEmbeddings(nn.Module):
    # TODO: input channel is yet to-be-decided (whether to use pre-encoding is not decided)
    def __init__(self, ksize = 3, out_channels = 128, input_channel = 4) -> None:
        super().__init__()
        self.initial_mapping = nn.Sequential(
            *makeMLP(input_channel, out_channels >> 2, bias = False)
        )
        
        self.convs = nn.Sequential(
            *makeConv1D(out_channels >> 2,  out_channels, ksize, norm = nn.BatchNorm1d(out_channels), padding = 0),
            *makeConv1D(out_channels,       out_channels, ksize, norm = nn.BatchNorm1d(out_channels), padding = 0),
        )

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        X = X.transpose(-1, -2)
        X = self.initial_mapping(X).transpose(-1, -2)
        X = self.convs(X)
        return X

class SeqPredictor(nn.Module):
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Parameter):
            nn.init.kaiming_normal_(m)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def __init__(self, args, emb_dim = 128) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        linear_dim = emb_dim << 2
        
        self.patch_embed = PatchEmbeddings(3, emb_dim, 4)
        self.avg_pool = nn.AdaptiveMaxPool1d(1)
        self.emb_drop = nn.Dropout(args.emb_dropout)
        self.drop_path = DropPath(args.path_dropout)
        
        self.input_layers = nn.Sequential(
            *makeConv1D(emb_dim,        linear_dim,     3, norm = nn.BatchNorm1d(linear_dim),   max_pool = 2),
            *makeConv1D(linear_dim,     linear_dim,     3, norm = nn.BatchNorm1d(linear_dim),   max_pool = 2),
            *makeConv1D(linear_dim,     linear_dim,     3, norm = nn.BatchNorm1d(linear_dim),   max_pool = 2,  act = None),
        )
        
        self.res_layers = nn.Sequential(
            *makeConv1D(linear_dim,     linear_dim,     3, norm = nn.BatchNorm1d(linear_dim)),
            *makeConv1D(linear_dim,     linear_dim,     3, norm = nn.BatchNorm1d(linear_dim)),
            *makeConv1D(linear_dim,     linear_dim,     3, norm = nn.BatchNorm1d(linear_dim),   act = None),
        )
        
        self.res_merge = nn.Sequential(
            *makeConv1D(linear_dim,     linear_dim,     3, norm = nn.BatchNorm1d(linear_dim)),
        )

        self.out_layers = nn.Sequential(
            *makeMLP(linear_dim,        linear_dim << 1,    args.class_dropout),
            *makeMLP(linear_dim << 1,   linear_dim,         args.class_dropout),
            nn.Linear(linear_dim, 2000),
        )
        # No sigmoid during classification (since there is one during AFL)
        self.apply(self.init_weight)

    def pad10(self, X: torch.Tensor) -> torch.Tensor:
        batch, total_len, channel = X.shape
        residual = total_len % self.win_size
        if residual:
            X = torch.cat((X, torch.zeros(batch, self.win_size - residual, channel, device = X.device)), dim = 1)
        return X

    def length_pad10(self, length: int) -> int:
        residual = length % self.win_size
        if residual:
            return length + self.win_size - residual
        return length

    def load(self, load_path:str, opt = None, other_stuff = None):
        save = torch.load(load_path)   
        save_model = save['model']                  
        state_dict = {k:save_model[k] for k in self.state_dict().keys()}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        if not opt is None:
            opt.load_state_dict(save['optimizer'])
        print("Swin Transformer 1D loaded from '%s'"%(load_path))
        if not other_stuff is None:
            return [save[k] for k in other_stuff]

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        # The input X is of shape (N, C, L) -> (batch, 4, seq_length)
        X = self.patch_embed(X)
        X = self.emb_drop(X)
        X = self.input_layers(X)
        tmp = self.res_layers(F.gelu(X))
        tmp = self.drop_path(X)
        X = F.gelu(X + tmp)
        X = self.res_merge(X)
        X = self.avg_pool(X).transpose(-1, -2)      # shape after avg pooling: (N, 1, C)
        return self.out_layers(X).squeeze(dim = 1)    # output is of shape (N, 2000)
    
if __name__ == "__main__":
    from utils.opt import get_opts
    opt_args = get_opts()
    stm = SeqPredictor(emb_dim = 96).cuda()
    test_seqs = torch.normal(0, 1, (8, 4, 1000)).cuda()
    result = stm.forward(test_seqs)
    