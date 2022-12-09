#-*-coding:utf-8-*-
"""
    PRML-Proj 4: Utility functions
    Generating easier-to-understand-and-load dataset from original .mtx and ATCG string txt
    @author: Qianyue He
    @date: 2022-11-16
"""

import os
import time
import tqdm
import torch
import shutil
import argparse
import numpy as np
from scipy.io import mmread
from datetime import datetime
from scipy.sparse import coo_matrix
from torch.utils.tensorboard import SummaryWriter

MAX_SEQ_LEN = 1500
MAX_ENCODING_LEN = 4
SQRT2 = 0.7071067811865476
INPUT_PREFIX = "../data_project4/"
INPUT_SUFFIX = "/sequences_"
LABEL_SUFFIX = "/matrix_"

MAPPING = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

ENCODING = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], dtype=np.uint8)

def convert_atcg_seqs(train: bool, out_prefix: str, seq_len: int):
    folder_name = "train" if train else "test"
    path = f"{INPUT_PREFIX}{folder_name}/{seq_len}{INPUT_SUFFIX}{folder_name}.txt"
    with open(path, 'r') as file:
        all_lines = file.readlines()
        seqs = np.zeros((MAX_ENCODING_LEN, seq_len), dtype = np.uint8)
        out_path_prefix = f"{out_prefix}{folder_name}_{seq_len}/seq_"
        for i, line in enumerate(tqdm.tqdm(all_lines)):
            na_lists = list(line)
            indices = list(map(lambda x: MAPPING[x], na_lists[:-1]))
            seqs[:, :seq_len] = ENCODING[indices].T
            seqs.tofile(f"{out_path_prefix}{i+1:05}.dat")

def convert_labels(train: bool, out_prefix: str, seq_len: int):
    folder_name = "train" if train else "test"
    path = f"{INPUT_PREFIX}{folder_name}/{seq_len}{LABEL_SUFFIX}{folder_name}.mtx"

    print(f"Start loading label matrix from '{path}'")
    start_time = time.time()
    matrix: coo_matrix = mmread(path)
    result = matrix.toarray().astype(np.uint8)
    del matrix
    print(f"Conversion completed. Time consumption: {time.time() - start_time:.3f} s. Shape: {result.shape}")

    for i, labels in enumerate(tqdm.tqdm(result)):
        out_path_prefix = f"{out_prefix}{folder_name}_{seq_len}_label/seq_"
        labels.tofile(f"{out_path_prefix}{i+1:05}.dat")

def save_model(model, path_info: dict, other_stuff: dict = None, opt = None):
    output_index = path_info['index'] % path_info['max_num'] + 1
    path = f"{path_info['dir']}chkpt_{output_index}_{path_info['type']}.{path_info['ext']}"
    checkpoint = {'model': model.state_dict(),}
    if not opt is None:
        checkpoint['optimizer'] = opt.state_dict()
    if not other_stuff is None:
        checkpoint.update(other_stuff)
    torch.save(checkpoint, path)
        
def get_summary_writer(exp_name: str, epochs:int, del_dir:bool):
    logdir = './logs/'
    if os.path.exists(logdir) and del_dir:
        shutil.rmtree(logdir)
    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-{1}-epoch{2}/".format(datetime.now(), exp_name, epochs)
    return SummaryWriter(log_dir = logdir + time_stamp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ================== ATCG converter args ==================
    parser.add_argument("--atcg_seq_len", type = int, default = 1000, help = "ATCG sequence length")
    parser.add_argument("--atcg_out_prefix", type = str, default = "../data/", help = "ATCG sequences folder path")
    parser.add_argument("--is_test", default = False, action = "store_true", help = "Is test datasets?")
    # ================== Label matrix converter args ===============
    parser.add_argument("--label_seq_id", type = int, default = 500, help = "ATCG sequence length")
    parser.add_argument("--label_out_prefix", type = str, default = "../data/", help = "ATCG sequences folder path")

    parser.add_argument("--cvt_label", action="store_true", default = False, help = "Whether to convert labels")

    args = parser.parse_args()
    if args.cvt_label:
        convert_labels(not args.is_test, args.label_out_prefix, args.label_seq_id)
    else:
        convert_atcg_seqs(not args.is_test, args.atcg_out_prefix, args.atcg_seq_len)
