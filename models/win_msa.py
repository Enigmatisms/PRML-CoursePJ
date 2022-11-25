#-*-coding:utf-8-*-
"""
    Swin Transformer Window-based multihead attention block
    This module is modified from my reproduction of Swin Transformer:
    Github Maevit: https://github.com/Enigmatisms/Maevit
    I am just reusing the Transformer Layer
    @author: Qianyue He
    @date: 2022-11-19
"""

import torch
from torch import nn
from typing import Optional
import matplotlib.pyplot as plt
from torch.nn import functional as F
from timm.models.layers import trunc_normal_

class WinMSA(nn.Module):
    def __init__(self, win_size = 10, emb_dim = 96, 
        head_num = 4,
        att_drop = 0.1, 
        proj_drop = 0.1, 
    ) -> None:
        super().__init__()
        self.win_size = win_size                    # window size is fixed = 10 (1D)
        self.emb_dim = emb_dim
        self.s = self.win_size >> 1                 # quote: displacing the window by floor(M / 2)
        
        self.emb_dim_h_k = emb_dim // head_num
        self.normalize_coeff = self.emb_dim_h_k ** (-0.5)
        self.head_num = head_num

        # positional embeddings
        self.positional_bias = nn.Parameter(torch.zeros(2 * win_size - 1, head_num), requires_grad = True)
        # using register buffer, this tensor will be moved to cuda device if .cuda() is called, also it is stored in state_dict
        self.register_buffer('relp_indices', WinMSA.getIndex(self.win_size))            

        self.qkv_attn = nn.Linear(emb_dim, emb_dim * 3, bias = True)
        self.proj_o = nn.Linear(emb_dim, emb_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(att_drop)
        trunc_normal_(self.positional_bias, std=.02)

    @staticmethod
    def getIndex(win_size:int) -> torch.LongTensor:
        ys, xs = torch.meshgrid(torch.arange(win_size), torch.arange(win_size), indexing = 'ij')
        indices = xs - ys + win_size - 1
        # print(indices)
        return indices

    def attention(self, X:torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, win_num, seq_len, _ = X.shape
        # typical input (batch_size, L / 10 (e.g., 500 / 10), 10, channel)
        # output (3, batch, win, head, seq, emb_dim/head_num) 0  1      2       3       4               5
        # actually, these are all the same since 2D image is already flattened
        # TODO: head num can not be devided (multiple)
        qkvs:torch.Tensor = self.qkv_attn(X).view(batch_size, win_num, seq_len, 3, self.head_num, self.emb_dim_h_k).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkvs[0], qkvs[1], qkvs[2]
        
        # 只有最后一个window存在不为0的attention mask（前一半与后一半不一样）
        # q @ k.T : shape (batch_size, win_num, head_num, seq, seq), att_mask added according to different window position  
        attn = q @ k.transpose(-1, -2) * self.normalize_coeff
        # print(self.relp_indices.shape, self.positional_bias.shape, seq_len, self.win_size, X.shape, attn.shape)

        position_bias = self.positional_bias[self.relp_indices.view(-1)].view(seq_len, seq_len, -1)     # here seq_len = 10
        position_bias = position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + position_bias.unsqueeze(0).unsqueeze(0)

        # attn = attn + self.positional_bias.view(self.head_num, -1)[:, self.relp_indices.view(-1)].view(self.head_num, seq_len, seq_len)
        if not mask is None:
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
        proba:torch.Tensor = F.softmax(attn, dim = -1)
        proba = self.attn_drop(proba)
        # proba: batch_size, window, head_num, seq_len, seq_len -> output (batch_size, win_num, head_num, seq_len, emb_dim/ head_num)
        proj_o = self.proj_o((proba @ v).transpose(2, 3).reshape(batch_size, win_num, seq_len, -1))
        return self.proj_drop(proj_o)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.attention(X, None)

# default invariant shift: win_size / 2
class SwinMSA(WinMSA):
    def __init__(self, atcg_len, win_size = 10, emb_dim = 96, head_num = 4) -> None:
        super().__init__(win_size, emb_dim, head_num)
        self.win_num = atcg_len // win_size
        # note that if win_size is odd, implementation will be the previous one, in which "half_att_size" is truely att_size / 2 
        self.register_buffer('att_mask', self.getAttentionMask())

    """
        shape of input tensor (batch_num, window_num, seq_length(number of embeddings in a window), emb_dim)
        output remains the same shape. Notice that, shift is already done outside of this class
    """
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.attention(X, self.att_mask)

    # somehow I think this is not so elegant, yet attention mask is tested
    def getAttentionMask(self) -> torch.Tensor:
        mask = torch.zeros(self.win_num, self.win_size, self.win_size)
        mask[-1, :, :] = -100 * torch.ones(self.win_size, self.win_size)
        
        """
            | A B | A, D is self attention. B, C is cross attention        
            | C D | B, C (cross att) should be prohibitted, therefore: add -100 proba bias
            The case of 1D attention (and shifting) is very simple
        """
        mask[-1, :self.s, :self.s] = 0.
        mask[-1, self.s:, self.s:] = 0.
        return mask

if __name__ == "__main__":
    print("Swin Multi-Head Attention unit test")
    sw = SwinMSA(30, 10, 1, 1).cuda()
    # print(sw.relp_indices.shape)
    mask = sw.getAttentionMask()
    for i in range(mask.shape[0]):
        plt.figure(i)
        plt.imshow(mask[i])
    plt.show()
