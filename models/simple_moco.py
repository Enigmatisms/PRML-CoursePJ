# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from torch.nn import functional as F
import random

def make_mlp(emb_dim = 256):
    return [
        nn.GELU(),
        nn.Linear(emb_dim, emb_dim, bias = False),
        nn.BatchNorm1d(emb_dim),
        nn.GELU(),
        nn.Linear(emb_dim, emb_dim, bias = False),
        nn.BatchNorm1d(emb_dim)
    ]

class SimpleEncoder(nn.Module):
    def __init__(self, emb_dim = 256):
        super().__init__()
        self.lin1 = nn.Sequential(*make_mlp(emb_dim))
        self.lin2 = nn.Sequential(*make_mlp(emb_dim))
        self.output = nn.Sequential(
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim >> 1, bias = False)
        )
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        tmp1 = self.lin1(X)
        X = tmp1 + X
        tmp2 = self.lin2(X)
        return self.output(X + tmp2)

class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder: SimpleEncoder, dim = 128, mlp_dim = 512, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 512)
        T: softmax temperature (default: 1.0)
        """
        super().__init__()

        self.T = T

        # build encoders
        self.base_encoder: SimpleEncoder = base_encoder()
        self.momentum_encoder: SimpleEncoder = base_encoder()
        
        self.predictor = MoCo._build_mlp(2, dim, mlp_dim, dim, False)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    @staticmethod
    def _build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(1, affine=False))

        return nn.Sequential(*mlp)

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = torch.arange(N, dtype=torch.long).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
    
    def load(self, load_path:str):
        save = torch.load(load_path)   
        save_model = save['model']                  
        state_dict = {k:save_model[k] for k in self.state_dict().keys()}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        print("MoCo base encoder loaded from '%s'"%(load_path))
    
def embedding_augmentation(X: torch.Tensor, p_param: dict) -> torch.Tensor:
    max_masking_len = p_param['max_masking_len']
    batch_size, length = X.shape    # (2000, 256)
    std_x = X.std(dim = 1) * p_param['std_scaler']  # noise level w.r.t to data std (std_scaler default 0.05)
    # random Gaussian noise
    X += torch.normal(0, std_x.unsqueeze(dim = -1).expand(-1, length)).cuda()
    if random.random() < p_param['masking_proba']:  # default 0.25
        mask = torch.ones(batch_size, length, device = X.device)
        mask_indices = torch.randint(0, length - 1, (batch_size, max_masking_len)).cuda()
        mask.scatter_(1, mask_indices, 0.)          # values will be masked
        X *= mask
    return X
    
if __name__ == "__main__":
    print("MoCo testing...")
    test_input1 = torch.normal(0, 1, (50, 256)).cuda()
    test_input2 = torch.normal(0, 1, (50, 256)).cuda()
    
    moco = MoCo(SimpleEncoder).cuda()
    result = moco.forward(test_input1, test_input2, 0.99)
    print(f"Result CL: {result}")