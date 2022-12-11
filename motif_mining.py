#-*-coding:utf-8-*-
"""
    Conv Predictor 1D training for Motif Mining: Simple Gradient Method
    The method is simple: set the input (requires_grad = True), then forward to get the loss
    and backward to get gradient of each input element
    @author: Qianyue He
    @date: 2022-11-25
    Milestone on @date 2022.12.8
"""
import tqdm
import torch
import scipy
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
    testset = CustomDataSet("./data/", atcg_len, transform, True, use_half = False)
    
    seq_model.eval()
    ret = {'model': seq_model, 'test_set': testset, 'args': args, 'loss_func': loss_func}
    
    return ret

def motif_mining_main(eval_kwargs):
    args        = eval_kwargs['args']
    testset     = eval_kwargs['test_set']   
    loss_func   = eval_kwargs['loss_func']
    model: SeqPredictor\
                = eval_kwargs['model']   

    test_set_len = len(testset)
    test_idxs = [10 * i for i in range(3)] if args.visualization else list(range(test_set_len))
    seq_num = len(test_idxs)
    
    resulting_grads = []
    max_seq_num = args.max_seq_num if args.max_seq_num > 0 else seq_num
    for i in tqdm.tqdm(range(max_seq_num)):
        idx = test_idxs[i]
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
    all_motifs = []
    for i in tqdm.tqdm(range(max_seq_num)):
        grads = resulting_grads[i]
        grads_filtered, peak_ids, peak_vals, trunc_mean, trunc_std = conditioned_peak_finder(grads)
        if args.visualization:
            if seq_num > 1:
                plt.subplot(seq_num, 1, i + 1)

            center = np.array([[0, trunc_mean], [args.atcg_len - 1, trunc_mean]])
            upper = center.copy()
            lower = center.copy()
            upper[:, -1] += trunc_std
            lower[:, -1] -= trunc_std
            plt.plot(xs, grads_filtered, label = f'sequence id = {test_idxs[i]}')
            plt.plot(center[:, 0], center[:, 1], linestyle = '--', label = 'mean')
            plt.plot(upper[:, 0], upper[:, 1], linestyle = '--', label = 'mean + std')
            plt.plot(lower[:, 0], lower[:, 1], linestyle = '--', label = 'mean - std')
            
            plt.scatter(xs, grads_filtered, s = 6)
            plt.scatter(peak_ids, peak_vals, s = 30, facecolors='none', edgecolors='r', label = 'motif peaks')
            plt.title(f'Extracted motifs ({len(peak_ids)}) in sequence {test_idxs[i]}')
            plt.ylabel('Gradient L2 norm')
            
            plt.grid(axis = 'both')
            plt.legend()
        else:
            sequence_str = testset.get_sequence(i)
            motifs = windowed_coarse_motif(sequence_str, peak_ids, args.motif_crop_size)
            all_motifs.extend(motifs)
    if args.visualization:
        plt.xlabel("sequence 'time step'")
        plt.show()

    print(f"Extracted motifs {len(all_motifs)}")
    motif_freq_analysis(all_motifs)

# I wish the number of peaks to be moderate (not too many and not too few)
def conditioned_peak_finder(x: np.ndarray, gaussian_sigma = 5., trunc_ratio = 0.15, min_max = [5, 15]):
    x = scipy.ndimage.filters.gaussian_filter1d(x, gaussian_sigma)
    trunc_size = int(x.shape[0] * trunc_ratio)
    peak_ids, trunc_mean, trunc_std = peak_finder(x, trunc_size, 1.0, True)
    num_peaks = len(peak_ids)
    if num_peaks < min_max[0]:
        peak_ids, trunc_mean, trunc_std = peak_finder(x, trunc_size, 1.0, False)            # lower prominence
    elif num_peaks > min_max[0]:        
        peak_ids, trunc_mean, trunc_std = peak_finder(x, trunc_size << 1, 1.0, True)        # better mean / std estimation
    peak_vals = x[peak_ids]
    return x, peak_ids, peak_vals, trunc_mean, trunc_std

def peak_finder(x: np.ndarray, truc_size = 50, prominence_scaler: float = 1.0, h_std_offset: bool = False):
    trunced: np.ndarray = x[truc_size:-truc_size]
    trunc_mean = trunced.mean()
    trunc_std = trunced.std()
    min_height = trunc_mean
    if h_std_offset: min_height += trunc_std
    
    peak_ids, _ = scipy.signal.find_peaks(x, prominence = trunc_std * prominence_scaler, height = [min_height, 1.])
    return peak_ids, trunc_mean, trunc_std

# I don't think the result is pretty accurate, therefore the function is named: coarse
def windowed_coarse_motif(sequence: str, centers: np.ndarray, window_size = 9):
    results = []
    half_w = window_size >> 1
    seq_len = len(sequence)
    for cid in centers:
        lower = max(0, cid - half_w)
        upper = min(seq_len, cid + half_w + 1)
        results.append(sequence[lower:upper])
    return results

def motif_freq_analysis(all_motifs: list):
    motif_dict = {}
    for motif in all_motifs:
        if motif not in motif_dict:
            motif_dict[motif] = 1
        else:
            motif_dict[motif] += 1
    entry_to_del = [key for key, value in motif_dict.items() if value < 20]
    print(f"After filtering, there are {len(entry_to_del)} entries to be removed.")
    for key in entry_to_del:
        del motif_dict[key]
    bars = list(motif_dict.values())
    plt.bar(list(range(len(motif_dict.keys()))), bars)
    plt.grid(axis = 'both')
    plt.show()

def main(context: dict):
    print("Motif mining ...")
    context['model'] = context['model'].cuda()
    motif_mining_main(context)

if __name__ == "__main__":
    parser = get_predictor_opts(True)
    parser.add_argument("-v", "--visualization", default = False, action = "store_true", help = "Visualize motif extraction")
    parser.add_argument("--motif_crop_size", type = int, default = 9, help = "Window size to crop out motif from the ATCG sequence")
    parser.add_argument("--max_seq_num", type = int, default = 2000, help = "Only test part of the sequences")
    opt_args = parser.parse_args()
    context = setup(opt_args)
    main(context)
    