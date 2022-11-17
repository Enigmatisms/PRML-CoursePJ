import numpy as np
import argparse
import tqdm
from scipy.io import mmread


MAX_SEQ_LEN = 1500
MAX_ENCODING_LEN = 4
SQRT2 = 0.7071067811865476
INPUT_PREFIX = "../data_project4/"
INPUT_SUFFIX = "/sequences_"

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

def convert_labels(path: str, out_path: str, start_num: int = 1):
    matrix = mmread(path)
    print(matrix)

# ==================== Mains ==========================
def convert_atcg_main(args):
    convert_atcg_seqs(args.is_train, args.atcg_out_prefix, args.atcg_seq_len)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ================== ATCG converter args ==================
    parser.add_argument("--atcg_seq_len", type = int, default = 1000, help = "ATCG sequence length")
    parser.add_argument("--atcg_out_prefix", type = str, default = "../data/", help = "ATCG sequences folder path")
    parser.add_argument("--is_train", default = True, action = "store_false", help = "Is train datasets?")
    # ================== Label matrix converter args ===============
    parser.add_argument("--label_in_file", type = str, default = "../data_project4/train/500/sequences_train.txt", help = "ATCG sequences file path")
    parser.add_argument("--label_out_path", type = str, default = "../data/train/", help = "ATCG sequences folder path")
    parser.add_argument("--label_start_id", type = int, default = 1, help = "ATCG sequence numbering starting id")

    args = parser.parse_args()

    convert_atcg_main(args)
    # convert_labels()
