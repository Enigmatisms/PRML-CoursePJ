#-*-coding:utf-8-*-
"""
    Conv Predictor 1D training for Motif Mining: Simple Gradient Method
    The method is simple: set the input (requires_grad = True), then forward to get the loss
    and backward to get gradient of each input element
    @author: Qianyue He
    @date: 2022-11-25
    Milestone on @date 2022.12.8
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from timm.loss import AsymmetricLossMultiLabel

from utils.opt import get_predictor_opts
from utils.dataset import CustomDataSet
from utils.train_helper import *
from models.seq_pred import SeqPredictor

default_chkpt_path = "./check_points/"
default_model_path = "./model/"
OUTPUT_DIM = 2000

def setup(args):
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benckmark = True
    
    # Bool options
    debugging           = args.debug
    
    # params for training
    
    # dataset specifications
    load_path           = args.load_path
    atcg_len            = args.atcg_len        
    
    # AsymmetricLossMultiLabel params
    asl_gamma_pos       = args.asl_gamma_pos
    asl_gamma_neg       = args.asl_gamma_neg
    asl_eps             = args.asl_eps
    asl_clip            = args.asl_clip
      
    if not torch.cuda.is_available():
        print("CUDA not available.")
        exit(-1)
    
    seq_model = SeqPredictor(args, emb_dim = 128)
    if not load_path:
        raise("LoadPathEmptyError: args.load_path is required in eval mode but not provided.")

    if debugging:
        for submodule in seq_model.modules():
            submodule.register_forward_hook(nan_hook)
        torch.autograd.set_detect_anomaly(True)

    # loss function adopts paper "Asymmetric Loss For Multi-Label Classification".
    loss_func = AsymmetricLossMultiLabel(gamma_pos = asl_gamma_pos, gamma_neg = asl_gamma_neg, clip = asl_clip, eps = asl_eps)
    transform = lambda x: x - 0.5
    testset = CustomDataSet("./data/", atcg_len, transform, False, use_half = False)
    
    seq_model.eval()
    ret = {'model': seq_model, 'test_set': testset, 'args': args, 'loss_func': loss_func}
    
    return ret

def eval(eval_kwargs):
    args        = eval_kwargs['args']
    testset     = eval_kwargs['test_set']   
    loss_func   = eval_kwargs['loss_func']
    model: SeqPredictor\
                = eval_kwargs['model']   

    test_idxs = [10 * i for i in range(4)]
    
    resulting_grads = []
    for idx in test_idxs:
        batch_x, batch_y = testset[idx]
        batch_x: torch.Tensor = batch_x.cuda().unsqueeze(0)
        batch_y: torch.Tensor = batch_y.cuda().unsqueeze(0)

        batch_x.requires_grad = True                # Motif Mining -- gradient requirement
        pred_y = model.forward(batch_x)
        loss: torch.Tensor = loss_func(pred_y, batch_y)
        loss.backward()
        
        grad_norm = batch_x.grad.norm(dim = -2).squeeze()
        resulting_grads.append(grad_norm.cpu().numpy())
    
    xs = np.arange(args.atcg_len)
    for i, grads in enumerate(resulting_grads):
        plt.subplot(4, 1, i + 1)
        plt.plot(xs, grads, label = f'sequence id = {test_idxs[i]}')
        plt.scatter(xs, grads, s = 7)
        plt.grid(axis = 'both')
        plt.legend()
    plt.savefig("test.png")
    


def main(context: dict):
    print("Motif mining ...")
    context['model'] = context['model'].cuda()
    eval(context)

if __name__ == "__main__":
    opt_args = get_predictor_opts()
    context = setup(opt_args)
    main(context)
    