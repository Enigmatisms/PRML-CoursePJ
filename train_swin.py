#-*-coding:utf-8-*-
"""
    Swin Transformer 1D training for Chromatin Openess Prediction
    Also, this will be pytorch template of mine in the future
    @author: Qianyue He
    @date: 2022-11-25
"""
import os
import tqdm
import torch

from timm.loss import AsymmetricLossMultiLabel

from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp.grad_scaler import GradScaler

from utils.opt import get_opts
from utils.train_helper import *
from models.swin import SwinTransformer
from utils.dataset import CustomDataSet
from utils.utils import get_summary_writer
from utils.cosine_anneal import LECosineAnnealingSmoothRestart

default_chkpt_path = "./check_points/"
default_model_path = "./model/"

def setup(args):
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benckmark = True
    
    # Bool options
    is_eval             = args.eval
    del_dir             = args.del_dir
    debugging           = args.debug
    
    # params for training
    weight_decay        = args.weight_decay
    
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
    
    swin_model = SwinTransformer(atcg_len, args)
    if os.path.exists(load_path):
        train_cnt, epoch = swin_model.load(load_path, opt, ["train_cnt", "epoch"])
    else:
        raise RuntimeError(f"Model file '{load_path}' does not exist.")

    if debugging:
        for submodule in swin_model.modules():
            submodule.register_forward_hook(nan_hook)
        torch.autograd.set_detect_anomaly(True)

    # loss function adopts paper "Asymmetric Loss For Multi-Label Classification".
    loss_func = AsymmetricLossMultiLabel(gamma_pos = asl_gamma_pos, gamma_neg = asl_gamma_neg, clip = asl_clip, eps = asl_eps)
    testset = CustomDataSet("./data/", atcg_len, None, False, True, args.half_opt)
    
    ret = {'model': swin_model, 'test_set': testset, 'args': args, 'loss_func': loss_func}
    if is_eval:
        trainset = None if is_eval else CustomDataSet("./data/", atcg_len, None, True, True, args.half_opt)
        opt = optim.AdamW(params = swin_model.parameters(), lr = args.lr, betas=(0.9, 0.999), weight_decay = weight_decay)
        lec_sch = LECosineAnnealingSmoothRestart(args)
        epochs = args.full_epochs + args.cooldown_epoch
        writer = get_summary_writer(epochs, del_dir)
        swin_model.eval()

        ret['opt']          = opt
        ret['epoch']        = epoch
        ret['writer']       = writer
        ret['opt_sch']      = lec_sch
        ret['train_set']    = trainset
        ret['train_cnt']    = train_cnt
    else:
        swin_model.train()
    return ret

def train(train_kwargs):
    pass

def eval(eval_kwargs):
    pass

def main(context: dict):
    pass

if __name__ == "__main__":
    opt_args = get_opts()
    context = setup(opt_args)
    main(context)
    