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
from utils.utils import get_summary_writer, save_model
from utils.cosine_anneal import LECosineAnnealingSmoothRestart

default_chkpt_path = "./check_points/"
default_model_path = "./model/"
OUTPUT_DIM = 2000

def setup(args):
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benckmark = True
    
    # Bool options
    is_eval             = args.eval
    del_dir             = args.del_dir
    debugging           = args.debug
    
    # params for training
    weight_decay        = args.adam_wdecay
    
    # dataset specifications
    load_path           = args.load_path
    atcg_len            = args.atcg_len        
    
    # AsymmetricLossMultiLabel params
    asl_gamma_pos       = args.asl_gamma_pos
    asl_gamma_neg       = args.asl_gamma_neg
    asl_eps             = args.asl_eps
    asl_clip            = args.asl_clip
    epoch               = 0
      
    if not torch.cuda.is_available():
        print("CUDA not available.")
        exit(-1)
    
    swin_model = SwinTransformer(atcg_len, args)
    if not load_path:
        if is_eval:
            raise("LoadPathEmptyError: args.load_path is required in eval mode but not provided.")
    else:
        load_path = os.path.join(default_chkpt_path if not args.load_model else default_model_path, args.load_path)
        if os.path.exists(load_path):
            epoch = swin_model.load(load_path, opt, ["epoch"])
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
        lec_sch = LECosineAnnealingSmoothRestart(args)
        opt = optim.AdamW(params = swin_model.parameters(), lr = lec_sch.lr(epoch), betas=(0.9, 0.999), weight_decay = weight_decay)
        epochs = args.full_epochs + args.cooldown_epochs
        writer = get_summary_writer(epochs, del_dir)
        swin_model.eval()

        ret['opt']          = opt
        ret['epoch']        = epoch
        ret['writer']       = writer
        ret['opt_sch']      = lec_sch
        ret['train_set']    = trainset
        ret['full_epoch']   = epochs
    else:
        swin_model.train()
    return ret

def train(train_kwargs):
    args        = train_kwargs['args']
    opt         = train_kwargs['opt']         
    epoch       = train_kwargs['epoch']       
    writer      = train_kwargs['writer']      
    trainset    = train_kwargs['train_set']   
    full_epoch  = train_kwargs['full_epoch'] 
    loss_func   = train_kwargs['loss_func']

    model: SwinTransformer\
                = train_kwargs['model']   
    lec_sch: LECosineAnnealingSmoothRestart\
                = train_kwargs['opt_sch']     

    train_loader    = DataLoader(trainset, args.batch_size, shuffle = True, num_workers = args.num_workers, drop_last = True)

    scaler = GradScaler() if args.half_opt else None
    
    loader_len = len(train_loader)
    for ep in tqdm.tqdm(range(epoch, full_epoch)):
        train_full_num = 0
        train_correct_num = 0
        total_loss = 0
        for i, (batch_x, batch_y) in enumerate(train_loader):
            opt.zero_grad()
            pred_y = model.forward(batch_x)

            # There might be preprocessing of predictions?
            if scaler is not None:
                with autocast():
                    loss = loss_func(pred_y, batch_y)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
            else:
                loss: torch.Tensor = loss_func(pred_y, batch_y)
                loss.backward()
                opt.step()
            total_loss += loss
            train_full_num += args.batch_size * OUTPUT_DIM
            train_correct_num += acc_calculate(pred_y, batch_y, args.pos_threshold)
            if args.train_verbose > 0 and i % args.train_verbose == 0:
                local_cnt = ep * loader_len + i
                local_acc = train_correct_num / OUTPUT_DIM
                writer.add_scalar('Loss/Train Loss', loss, local_cnt)
                writer.add_scalar('Acc/Train Acc', local_acc, local_cnt)

                print(f"Traning Epoch: {ep:4d} / {full_epoch:4d}\
                    \tbatch: {i:3d} / {loader_len:3d}\
                    \ttrain loss: {loss.item():.5f}\
                    \ttrain acc: {local_acc:.4f}"
                )
        opt, current_lr = lec_sch.update_opt_lr(ep, opt)
        vanilla_acc = train_correct_num / train_full_num
        total_loss /= loader_len

        print(f"Traning Epoch: {ep:4d} / {full_epoch:4d}\ttrain loss: {total_loss.item():.5f}\ttrain acc: {vanilla_acc:.4f}\tlearing rate: {current_lr:.7f}")        
        writer.add_scalar('Loss/Train Avg Loss', total_loss, ep)
        writer.add_scalar('Acc/Train Avg Acc', vanilla_acc, ep)

        chkpt_info = {'index': ep, 'max_num': 3, 'dir': default_chkpt_path, 'type': 'baseline', 'ext': 'pt'}
        save_model(model, chkpt_info, {'epoch': ep}, opt)

        if ep % args.train_eval_time == 0 and ep > epoch:
            eval(train_kwargs, ep)
    print("Training completed.")
    model_info = {'index': ep, 'max_num': 3, 'dir': default_chkpt_path, 'type': 'baseline', 'ext': 'pt'}
    save_model(model, model_info, opt)

def eval(eval_kwargs, cur_epoch = 0, use_writer = True):
    args        = eval_kwargs['args']
    writer      = eval_kwargs['writer']      
    testset     = eval_kwargs['test_set']   
    loss_func   = eval_kwargs['loss_func']
    model: SwinTransformer\
                = eval_kwargs['model']   

    test_loader = DataLoader(testset, args.test_batch_size, shuffle = True, num_workers = args.num_workers, drop_last = True)
    test_batches = args.test_batches if args.test_batches else len(test_loader)

    scaler = GradScaler() if args.half_opt else None
    test_full_num = 0
    test_correct_num = 0
    total_loss = 0
    for i, (batch_x, batch_y) in enumerate(test_loader):
        if i >= test_batches: break
        pred_y = model.forward(batch_x)
        if scaler is not None:
            with autocast():
                loss = loss_func(pred_y, batch_y)
        else:
            loss: torch.Tensor = loss_func(pred_y, batch_y)
        test_full_num += args.test_batch_size * OUTPUT_DIM
        test_correct_num += acc_calculate(pred_y, batch_y, args.pos_threshold)
        total_loss += loss
    total_loss /= test_batches
    vanilla_acc = test_correct_num / test_full_num
    print(f"Evaluating Epoch: {cur_epoch:4d}\ttest loss: {total_loss.item():.5f}\ttest acc: {vanilla_acc:.4f}\t")    
    if use_writer:    
        writer.add_scalar('Loss/Test Avg Loss', total_loss, cur_epoch)
        writer.add_scalar('Acc/Test Avg Acc', vanilla_acc, cur_epoch)

def main(context: dict):
    if "trainset" in context:
        train(context)
    else:
        eval(context, 0, False)

if __name__ == "__main__":
    opt_args = get_opts()
    context = setup(opt_args)
    main(context)
    