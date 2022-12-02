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
import einops
from torch import nn
from models.win_msa import WinMSA, SwinMSA
from timm.models.layers.drop import DropPath

def makeMLP(in_chan, mlp_dropout):
    return nn.Sequential(
        nn.Linear(in_chan, 4 * in_chan),
        nn.GELU(),
        nn.Dropout(mlp_dropout),
        nn.Linear(4 * in_chan, in_chan),
        nn.Dropout(mlp_dropout)
    )

class SwinTransformerLayer(nn.Module):
    # emb_dim should be the integer multiple of 16 (which will be fine)
    # notice that, atcg_len here is actually modified (according to each layer, after merging and padding)
    def __init__(self, atcg_len, win_size = 10, emb_dim = 96, head_num = 4, mlp_dropout=0.1, path_drop=0.1, patch_merge = False) -> None:
        super().__init__()
        self.win_size = win_size
        self.emb_dim = emb_dim
        self.atcg_len = atcg_len
        self.win_num = atcg_len // win_size
        
        self.merge_lin = nn.Linear(emb_dim << 1, emb_dim * 3 // 2)      # 1.5
        self.drop_path = DropPath(path_drop)
        self.pre_ln = nn.LayerNorm(emb_dim)
        self.post_ln = nn.LayerNorm(emb_dim)
        self.head_num = head_num
        self.mlp = makeMLP(emb_dim, mlp_dropout)
        self.win_msa = WinMSA(win_size, emb_dim, head_num)
        self.swin_msa = SwinMSA(atcg_len, win_size, emb_dim, head_num)
        self.merge_ln = None
        self.patch_merge = patch_merge
        if patch_merge == True:
            self.merge_ln = nn.LayerNorm(emb_dim << 1)

    def merge(self, X:torch.Tensor)->torch.Tensor:
        # input shape (N, win_num, L, C), L is actually 10 here
        X = einops.rearrange(X, 'N win_l L C -> N (win_l L) C', win_l = self.win_num)
        X = einops.rearrange(X, 'N (L m) C -> N L (m C)', m = 2)
        X = self.merge_lin(self.merge_ln(X))
        return X    

    def layerForward(self, X:torch.Tensor, use_swin = False) -> torch.Tensor:
        tmp = self.pre_ln(X)
        tmp = self.swin_msa(tmp) if use_swin else self.win_msa(tmp)
        X = self.post_ln(X + self.drop_path(tmp))
        tmp2 = self.mlp(X)
        return X + self.drop_path(tmp2)

    # (N, win_num, L, C) -> (N, win_num, L, C) or (N, win_num/2, L, C)
    def forward(self, X:torch.Tensor)->torch.Tensor:
        # patch partion is done in every layer
        X = einops.rearrange(X, 'N (m L) C -> N m L C', L = self.win_size)       
        X = self.layerForward(X)
        # shifting, do not forget this
        X = torch.roll(X, shifts = -(self.win_size >> 1), dims = -2)
        X = self.layerForward(X, use_swin = True)
        # inverse shifting procedure
        X = torch.roll(X, shifts = self.win_size >> 1, dims = -2)
        # after attention op, tensor must be reshape back to the original shape
        if self.patch_merge:
            return self.merge(X)
        else:
            X = einops.rearrange(X, 'N m L C -> N (m L) C')       
        return X

"""
    To my suprise, patching embedding generation in official implementation contains Conv2d... I thought this was conv-free
    Patch partition and embedding generation in one shot, in the paper, it is said that:
    'A linear embedding layer is applied on this raw-valued feature to project it to an arbitrary dimension'
"""
class PatchEmbeddings(nn.Module):
    def makeConv1D(in_chan, out_chan, ksize = 3, act = nn.GELU(), norm = None):
        layer = [nn.Conv1d(in_chan, out_chan, kernel_size = ksize, padding = ksize >> 1)]
        if norm is not None:
            layer.append(norm)
        if act is not None:
            layer.append(act)
        return layer
    
    # TODO: input channel is yet to-be-decided (whether to use pre-encoding is not decided)
    def __init__(self, ksize = 3, win_size = 10, out_channels = 48, input_channel = 4, norm_layer = None) -> None:
        super().__init__()
        self.win_size = win_size
        
        self.convs = nn.Sequential(
            *PatchEmbeddings.makeConv1D(input_channel, out_channels >> 1, ksize),
            *PatchEmbeddings.makeConv1D(out_channels >> 1, out_channels, ksize),
            *PatchEmbeddings.makeConv1D(out_channels, out_channels, ksize, norm = norm_layer),
        )

    # TODO: dataset is not correct (C should be dim0, L should be dim1)
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        X = self.convs(X)
        X = X.permute(0, 2, 1)      # (N, C, L) to (N, L, C)
        return X
        # output x is (N, win_num, win_size (typically 10), C)

class SwinTransformer(nn.Module):
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Parameter):
            nn.init.kaiming_normal_(m)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def __init__(self, atcg_len, args, win_size = 10, emb_dim = 96, 
        head_num = (3, 6, 12, 18),
    ) -> None:
        super().__init__()
        self.win_size = win_size
        self.emb_dim = emb_dim
        self.atcg_len = atcg_len
        self.head_num = head_num
        current_img_size = atcg_len
        
        self.patch_embed = PatchEmbeddings(3, win_size, emb_dim, 4)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.emb_drop = nn.Dropout(args.emb_dropout)
        # input image_size / 4, output_imgae_size / 4
        self.swin_layers = nn.ModuleList([])
        # 96 144 216 324 -> 486
        num_layers = (2, 2, 4, 2)
        
        for i, num_layer in enumerate(num_layers):
            num_layer = num_layers[i]
            num_head = head_num[i]
            for _ in range(num_layer - 1):
                self.swin_layers.append(SwinTransformerLayer(current_img_size, win_size, emb_dim, num_head, args.mlp_dropout, args.path_dropout))

            # This should be more focused
            self.swin_layers.append(SwinTransformerLayer(current_img_size, win_size, emb_dim, num_head, args.mlp_dropout, args.path_dropout, True))
            current_img_size >>= 1
            current_img_size = self.length_pad10(current_img_size)
            emb_dim = emb_dim // 2 * 3
        # final channel 768, maybe it is too narrow
        self.classify = nn.Sequential(
            nn.Linear(emb_dim, emb_dim << 1),
            nn.Dropout(args.class_dropout),
            nn.ReLU(),
            nn.Linear(emb_dim << 1, 2000),
            nn.Dropout(args.class_dropout),
            nn.ReLU(),
            nn.Linear(2000, 2000),
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
            return length + residual
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
        batch_size, _, _ = X.shape 
        X = self.patch_embed(X)
        X = self.emb_drop(X)
        for _, layer in enumerate(self.swin_layers):
            X = self.pad10(X)       # should be tested (if win_num is the multiple of 10, then nothing is padded)
            X = layer(X)
        channel_num = X.shape[-1]
        X = X.view(batch_size, -1, channel_num).transpose(-1, -2)
        X = self.avg_pool(X).transpose(-1, -2)      # shape after avg pooling: (N, 1, C)
        return self.classify(X).squeeze(dim = 1)    # output is of shape (N, 2000)
    
if __name__ == "__main__":
    print("Swin Transformer 1D unit test")
    SEQ_LEN = 1000
    stm = SwinTransformer(SEQ_LEN, win_size = 10, emb_dim = 96).cuda()
    test_seqs = torch.normal(0, 1, (8, 4, SEQ_LEN)).cuda()
    result = stm.forward(test_seqs)
    