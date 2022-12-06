#-*-coding:utf-8-*-
"""
    Swin Transformer Layer implementation
    This module is modified from my reproduction of Swin Transformer:
    Github Maevit: https://github.com/Enigmatisms/Maevit
    I am just reusing the Swin Transformer Layer as SwinTransformer1D
    @author: Qianyue He
    @date: 2022-11-19
"""

import torch
from torch import nn
    
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
    def __init__(self, ksize = 3, ksize_init = 3, out_channels = 128, input_channel = 4, layer_dropout = 0.1) -> None:
        super().__init__()
        self.initial_mapping = nn.Linear(input_channel, out_channels >> 2, bias = False)
        
        self.convs = nn.Sequential(
            *makeConv1D(out_channels >> 2, out_channels >> 1, ksize_init, norm = nn.BatchNorm1d(out_channels >> 1), padding = 0),
            *makeConv1D(out_channels >> 1, out_channels, ksize, norm = nn.Dropout(layer_dropout), padding = 0),
            *makeConv1D(out_channels, out_channels, ksize, norm = nn.Dropout(layer_dropout), padding = 0),
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
            nn.init.constant_(m.bias, 0)
    def __init__(self, args, emb_dim = 128) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        linear_dim = emb_dim << 2
        
        self.patch_embed = PatchEmbeddings(3, 17, emb_dim, 4, args.layer_dropout)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.emb_drop = nn.Dropout(args.emb_dropout)
        self.conv_layer1 = nn.Sequential(
            *makeConv1D(emb_dim,        emb_dim << 1,   3, norm = nn.Dropout(args.layer_dropout),   max_pool = 2),
            *makeConv1D(emb_dim << 1,   linear_dim,     3, norm = nn.Dropout(args.layer_dropout),   max_pool = 2),
            *makeConv1D(linear_dim,     linear_dim,     3, norm = nn.Dropout(args.layer_dropout),   max_pool = 2),
        )
        self.conv_layer2 = nn.Sequential(
            *makeConv1D(linear_dim,     linear_dim,     3, norm = nn.Dropout(args.layer_dropout),   max_pool = 2),
            *makeConv1D(linear_dim,     linear_dim,     3, norm = nn.Dropout(args.layer_dropout),   max_pool = 2),
            *makeConv1D(linear_dim,     linear_dim,     3, norm = nn.Dropout(args.layer_dropout)),
        )

        self.classify = nn.Sequential(
            nn.Dropout(args.class_dropout),
            nn.Linear(linear_dim, linear_dim << 1),
            nn.GELU(),
            nn.Dropout(args.class_dropout),
            nn.Linear(linear_dim << 1, linear_dim),
            nn.GELU(),
            nn.Linear(linear_dim, 2000),
        )
        # No sigmoid during classification (since there is one during AFL)
        self.apply(self.init_weight)

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
        X = self.conv_layer1(X)
        X = self.conv_layer2(X)
        X = self.avg_pool(X).transpose(-1, -2)      # shape after avg pooling: (N, 1, C)
        return self.classify(X).squeeze(dim = 1)    # output is of shape (N, 2000)
    
if __name__ == "__main__":
    print("Swin Transformer 1D unit test")
    SEQ_LEN = 1000
    stm = SeqPredictor(emb_dim = 96).cuda()
    test_seqs = torch.normal(0, 1, (8, 4, SEQ_LEN)).cuda()
    result = stm.forward(test_seqs)
    