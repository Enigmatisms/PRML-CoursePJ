#-*-coding:utf-8-*-
"""
    PRML-Proj 4: Utility functions
    Generating easier-to-understand-and-load dataset from original .mtx and ATCG string txt
    @author: Qianyue He
    @date: 2022-11-16
"""

import numpy as np
import argparse
import tqdm
import time
from scipy.io import mmread
from scipy.sparse import coo_matrix

MAX_SEQ_LEN = 1500
MAX_ENCODING_LEN = 4
SQRT2 = 0.7071067811865476
INPUT_PREFIX = "../data_project4/"
INPUT_SUFFIX = "/sequences_"
LABEL_SUFFIX = "/matrix_"

MAPPING = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
ENCODING_BIG = np.array([
    [1, 0, 0, 0, SQRT2, 0, SQRT2, 0, 0.5, -0.5, -0.5, 0.5, 1],
    [0, 1, 0, 0, 0, SQRT2, 0, SQRT2, 0.5, 0.5, 0.5, 0.5, 1],
    [0, 0, 1, 0, SQRT2, 0, -SQRT2, 0, -0.5, -0.5, 0.5, 0.5, 1],
    [0, 0, 0, 1, 0, SQRT2, 0, -SQRT2, -0.5, 0.5, -0.5, 0.5, 1],
])

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
        seqs = np.zeros((seq_len, MAX_ENCODING_LEN), dtype = np.uint8)
        out_path_prefix = f"{out_prefix}{folder_name}_{seq_len}/seq_"
        for i, line in enumerate(tqdm.tqdm(all_lines)):
            na_lists = list(line)
            indices = list(map(lambda x: MAPPING[x], na_lists[:-1]))
            length = len(indices)
            seqs[:length, :] = ENCODING[indices]
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

# ==================== Mains ==========================
def convert_atcg_main(args):
    convert_atcg_seqs(not args.is_test, args.atcg_out_prefix, args.atcg_seq_len)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ================== ATCG converter args ==================
    parser.add_argument("--atcg_seq_len", type = int, default = 1000, help = "ATCG sequence length")
    parser.add_argument("--atcg_out_prefix", type = str, default = "../data/", help = "ATCG sequences folder path")
    parser.add_argument("--is_test", default = False, action = "store_true", help = "Is test datasets?")
    # ================== Label matrix converter args ===============
    parser.add_argument("--label_seq_id", type = int, default = 500, help = "ATCG sequence length")
    parser.add_argument("--label_out_prefix", type = str, default = "../data/", help = "ATCG sequences folder path")

    args = parser.parse_args()
    # convert_atcg_main(args)
    convert_labels(not args.is_test, args.label_out_prefix, args.label_seq_id)
