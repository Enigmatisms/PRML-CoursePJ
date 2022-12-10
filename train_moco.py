#-*-coding:utf-8-*-
"""
    To boost clustering result, I leveraged contrastive learning here: MoCo-v3 
    @author: Qianyue He
    @date: 2022-12-10
"""
import os
import tqdm
import torch

from torch import optim
from random import choices
from torch.cuda.amp import autocast as autocast

from utils.opt import get_moco_opts
from utils.train_helper import *
from utils.utils import get_summary_writer, save_model
from utils.cosine_anneal import LECosineAnnealingSmoothRestart

from models.seq_pred import get_wd_params
from models.simple_moco import MoCo, SimpleEncoder, embedding_augmentation

from spectral_clustering import get_model_embeddings

default_chkpt_path = "./check_points/"
default_model_path = "./model/"
OUTPUT_DIM = 2000

def setup(args):
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    
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
    epoch               = 0
      
    if not torch.cuda.is_available():
        print("CUDA not available.")
        exit(-1)
    
    moco = MoCo(SimpleEncoder).cuda()
    if not load_path:
        if is_eval:
            raise("LoadPathEmptyError: args.load_path is required in eval mode but not provided.")
    else:
        load_path = os.path.join(default_chkpt_path if not args.load_model else default_model_path, args.load_path)
        if os.path.exists(load_path):
            epoch = moco.load(load_path, None, ["epoch"])
        else:
            raise RuntimeError(f"Model file '{load_path}' does not exist.")

    if debugging:
        for submodule in moco.modules():
            submodule.register_forward_hook(nan_hook)
        torch.autograd.set_detect_anomaly(True)

    seq_model_path = f"{args.seq_model_dir}{args.seq_model_name}{atcg_len}.pt"
    train_set = get_model_embeddings(seq_model_path, True).cuda()                   # (2000, 256)
    if args.no_split == False:
        test_set = train_set[-100:, ...]
        train_set = train_set[:-100, ...]
    else:
        test_set = train_set
    
    ret = {'model': moco, 'test_set': test_set, 'args': args}
    
    if is_eval:
        moco.eval()
    else:
        lec_sch = LECosineAnnealingSmoothRestart(args)
        # Treat weight / bias / Batch norm params differently in terms of weight decay!
        decay_params, no_decay_params = get_wd_params(moco)
        decay_group = {'params': decay_params, 'weight_decay': weight_decay, 'lr': lec_sch.lr(epoch), 'betas': (0.9, 0.999)}
        no_decay_group = {'params': no_decay_params, 'weight_decay': 0., 'lr': lec_sch.lr(epoch), 'betas': (0.9, 0.999)}
        opt = optim.AdamW([decay_group, no_decay_group])
        epochs = args.full_epochs + args.cooldown_epochs + args.warmup_epochs
        writer = get_summary_writer(args.exp_name, epochs, del_dir)
        aug_dict = {'max_masking_len': args.max_masking_len, 'std_scaler': args.std_scaler, 'masking_proba': args.masking_proba}

        ret['opt']          = opt
        ret['epoch']        = epoch
        ret['writer']       = writer
        ret['opt_sch']      = lec_sch
        ret['train_set']    = train_set
        ret['full_epoch']   = epochs
        ret['aug_dict']     = aug_dict
        moco.train()
    return ret

def train(train_kwargs):
    args        = train_kwargs['args']
    opt         = train_kwargs['opt']         
    epoch       = train_kwargs['epoch']       
    writer      = train_kwargs['writer']      
    trainset    = train_kwargs['train_set']   
    full_epoch  = train_kwargs['full_epoch'] 
    model: MoCo = train_kwargs['model']   

    lec_sch: LECosineAnnealingSmoothRestart\
                = train_kwargs['opt_sch']     

    model = model.cuda()

    total_num_samples = trainset.shape[0]         # 2000 or 1900 (with split)
    num_batches = total_num_samples // args.batch_size
    for ep in tqdm.tqdm(range(epoch, full_epoch), position=0, leave=True):
        total_loss = 0
        for i in range(num_batches):
            indices = choices(range(total_num_samples), k = args.batch_size)
            embedding_1 = trainset[indices].cuda()                              # original data
            embedding_2 = embedding_augmentation(embedding_1, train_kwargs['aug_dict'])         # augmented positives
            opt.zero_grad()
            # There might be preprocessing of predictions?
            if args.half_opt:
                with autocast():
                    contrastive_loss = model.forward(embedding_1, embedding_2, args.moco_m)
            else:
                contrastive_loss = model.forward(embedding_1, embedding_2, args.moco_m)
            contrastive_loss.backward()
            opt.step()
            total_loss += contrastive_loss
            if args.train_verbose > 0 and i % args.train_verbose == 0:
                local_cnt = ep * num_batches + i
                writer.add_scalar('Loss/Train Loss', contrastive_loss, local_cnt)

                print(f"Traning Epoch: {ep:4d} / {full_epoch:4d}\
                    \tbatch: {i:3d} / {num_batches:3d}\
                    \ttrain loss: {contrastive_loss.item():.5f}"
                )
        opt, current_lr = lec_sch.update_opt_lr(ep, opt)
        total_loss /= total_num_samples

        print(f"Traning Epoch (Pro): {ep:4d} / {full_epoch:4d}\ttrain loss: {total_loss.item():.5f}\tlearing rate: {current_lr:.7f}")        
        writer.add_scalar('Loss/Train Avg Loss', total_loss, ep)
        writer.add_scalar('Learning rate', current_lr, ep)

        chkpt_info = {'index': ep, 'max_num': 2, 'dir': default_chkpt_path, 'type': f'{args.exp_name}_{args.atcg_len}', 'ext': 'pt'}
        save_model(model, chkpt_info, {'epoch': ep}, opt)

        if ep % args.train_eval_time == 0:
            eval(train_kwargs, ep, resume = True)
    print("Training completed.")
    model_info = {'index': ep, 'max_num': 2, 'dir': default_model_path, 'type': f'{args.exp_name}_{args.atcg_len}', 'ext': 'pt'}
    save_model(model, model_info, opt = opt)

def eval(eval_kwargs, cur_epoch = 0, use_writer = True, resume = False):
    args        = eval_kwargs['args']
    testset     = eval_kwargs['test_set']   
    model: MoCo = eval_kwargs['model']   

    test_set_total_len = testset.shape[0] 
    test_set_batches = test_set_total_len // args.test_batch_size
    total_loss = 0
    if resume:
        model.eval()
    with torch.no_grad():
        for _ in range(test_set_batches):
            indices = choices(range(test_set_total_len), k = args.test_batch_size)
            embedding_1 = testset[indices].cuda()                                               # original data
            embedding_2 = embedding_augmentation(embedding_1, eval_kwargs['aug_dict'])          # augmented positives
            
            # Do not update params during evaluation
            if args.half_opt:
                with autocast():
                    contrastive_loss = model.forward(embedding_1, embedding_2, 1.)
            else:
                contrastive_loss = model.forward(embedding_1, embedding_2, 1.)
                
            total_loss += contrastive_loss
    if resume:
        model.train()
    total_loss /= test_set_total_len
    print(f"Evaluating Epoch: {cur_epoch:4d}\ttest loss: {total_loss.item():.5f}")    
    if use_writer:    
        eval_kwargs['writer'].add_scalar('Loss/Test Avg Loss', total_loss, cur_epoch)

def main(context: dict):
    if "train_set" in context:
        print("Momentum Contrast training...")
        train(context)
    else:
        print("Momentum Contrast evaluating...")
        context['model'] = context['model'].cuda()
        eval(context, 0, False, False, auc = True)

if __name__ == "__main__":
    opt_args = get_moco_opts()
    context = setup(opt_args)
    main(context)
    