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
    
    swin_model = SwinTransformer(atcg_len, args, emb_dim = 96)
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
    transform = lambda x: x - 0.5
    testset = CustomDataSet("./data/", atcg_len, transform, False, args.half_opt)
    
    ret = {'model': swin_model, 'test_set': testset, 'args': args, 'loss_func': loss_func}
    if is_eval:
        swin_model.eval()
    else:
        trainset = None if is_eval else CustomDataSet("./data/", atcg_len, transform, True, args.half_opt)
        lec_sch = LECosineAnnealingSmoothRestart(args)
        opt = optim.RMSprop(params = swin_model.parameters(), lr = lec_sch.lr(epoch))
        epochs = args.full_epochs + args.cooldown_epochs
        writer = get_summary_writer(epochs, del_dir)

        ret['opt']          = opt
        ret['epoch']        = epoch
        ret['writer']       = writer
        ret['opt_sch']      = lec_sch
        ret['train_set']    = trainset
        ret['full_epoch']   = epochs
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
    model = model.cuda()

    scaler = GradScaler() if args.half_opt else None
    
    loader_len = len(train_loader)
    for ep in tqdm.tqdm(range(epoch, full_epoch)):
        train_full_num = 0
        train_correct_num = 0
        total_loss = 0
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            opt.zero_grad()
            # There might be preprocessing of predictions?
            if scaler is not None:
                with autocast():
                    pred_y = model.forward(batch_x)
                    loss = loss_func(pred_y, batch_y)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
            else:
                pred_y = model.forward(batch_x)
                loss: torch.Tensor = loss_func(pred_y, batch_y)
                loss.backward()
                opt.step()
            total_loss += loss
            local_correct_num, total_num, all_classes = acc_calculate(pred_y.detach(), batch_y, args.pos_threshold)
            train_correct_num += local_correct_num
            train_full_num += total_num
            if args.train_verbose > 0 and i % args.train_verbose == 0:
                local_cnt = ep * loader_len + i
                local_acc = local_correct_num / total_num
                local_full_acc = all_classes / (OUTPUT_DIM * args.batch_size)
                writer.add_scalar('Loss/Train Loss', loss, local_cnt)
                writer.add_scalar('Acc/Train Acc', local_acc, local_cnt)
                writer.add_scalar('Acc/Train Acc (All)', local_full_acc, local_cnt)

                print(f"Traning Epoch: {ep:4d} / {full_epoch:4d}\
                    \tbatch: {i:3d} / {loader_len:3d}\
                    \ttrain loss: {loss.item():.5f}\
                    \ttrain acc: {local_acc:.4f}\
                    \ttrain acc full: {local_full_acc:.4f}"
                )
        opt, current_lr = lec_sch.update_opt_lr(ep, opt)
        vanilla_acc = train_correct_num / train_full_num
        total_loss /= loader_len

        print(f"Traning Epoch: {ep:4d} / {full_epoch:4d}\ttrain loss: {total_loss.item():.5f}\ttrain acc: {vanilla_acc:.4f}\tlearing rate: {current_lr:.7f}")        
        writer.add_scalar('Loss/Train Avg Loss', total_loss, ep)
        writer.add_scalar('Acc/Train Avg Acc', vanilla_acc, ep)
        writer.add_scalar('Learning rate', current_lr, ep)

        chkpt_info = {'index': ep, 'max_num': 2, 'dir': default_chkpt_path, 'type': f'baseline_{args.atcg_len}', 'ext': 'pt'}
        save_model(model, chkpt_info, {'epoch': ep}, opt)

        if ep % args.train_eval_time == 0:
            eval(train_kwargs, ep, resume = True)
    print("Training completed.")
    model_info = {'index': ep, 'max_num': 2, 'dir': default_model_path, 'type': f'baseline_{args.atcg_len}', 'ext': 'pt'}
    save_model(model, model_info, opt)

def eval(eval_kwargs, cur_epoch = 0, use_writer = True, resume = False):
    args        = eval_kwargs['args']
    testset     = eval_kwargs['test_set']   
    loss_func   = eval_kwargs['loss_func']
    model: SwinTransformer\
                = eval_kwargs['model']   

    test_loader = DataLoader(testset, args.test_batch_size, shuffle = False, num_workers = 2, drop_last = False)
    test_batches = args.test_batches if args.test_batches else len(test_loader)

    scaler = GradScaler() if args.half_opt else None
    target_pos_num = 0
    pred_pos_num = 0
    test_full_num = 0
    total_loss = 0
    if resume:
        model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            if i >= test_batches: break
            if scaler is not None:
                with autocast():
                    pred_y = model.forward(batch_x)
                    loss = loss_func(pred_y, batch_y)
            else:
                pred_y = model.forward(batch_x)
                loss: torch.Tensor = loss_func(pred_y, batch_y)
            local_correct_num, batch_pos_num, full_num = acc_calculate(pred_y.detach(), batch_y, args.pos_threshold)
            target_pos_num += batch_pos_num
            pred_pos_num += local_correct_num
            total_loss += loss
            test_full_num += full_num
    if resume:
        model.train()
    total_loss /= test_batches
    vanilla_acc = test_full_num / (test_batches * args.test_batch_size * OUTPUT_DIM)
    vanilla_pos_acc = pred_pos_num / target_pos_num
    print(f"Evaluating Epoch: {cur_epoch:4d}\ttest loss: {total_loss.item():.5f}\ttest acc: {vanilla_pos_acc:.4f}\ttest acc (All): {vanilla_acc:.4f}")    
    if use_writer:    
        eval_kwargs['writer'].add_scalar('Loss/Test Avg Loss', total_loss, cur_epoch)
        eval_kwargs['writer'].add_scalar('Acc/Test Avg Acc', vanilla_pos_acc, cur_epoch)
        eval_kwargs['writer'].add_scalar('Acc/Test Avg Acc (All)', vanilla_acc, cur_epoch)

def main(context: dict):
    if "train_set" in context:
        print("Swin Transformer 1D training...")
        train(context)
    else:
        print("Swin Transformer 1D evaluating...")
        context['model'] = context['model'].cuda()
        eval(context, 0, False)

if __name__ == "__main__":
    opt_args = get_opts()
    context = setup(opt_args)
    main(context)
    