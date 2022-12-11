#-*-coding:utf-8-*-
"""
    PRML-Proj 4: Custom dataset for pytorch loader: parallel worker loading
    @author: Qianyue He
    @date: 2022-11-17
"""

import os
import torch
import natsort
import numpy as np
from random import randint
from torch.utils import data

MAPPING = np.array(['A', 'T', 'C', 'G'])
        
class CustomDataSet(data.Dataset):
    """
        root_dir: dataset root directory, should end with "/"
    """
    def __init__(self, root_dir: str, seq_num: int, transform = None, is_train = True, use_half = False, mix_up = 0.0):
        self.is_train = is_train
        self.root_dir = root_dir
        train_prefix = "train" if is_train else "test"

        self.seq_num = seq_num
        self.data_dir = f"{root_dir}{train_prefix}_{seq_num}/"
        self.label_dir = f"{root_dir}{train_prefix}_{seq_num}_label/"

        self.transform = transform
        img_names = filter(lambda x: x.endswith("dat"), os.listdir(self.data_dir))      # file name inside data_dir is the same with those in label_dir
        all_ndarray = [name for name in img_names]
        self.names = natsort.natsorted(all_ndarray)
        self.data_num = len(self.names)
        
        self.use_half = use_half
        self.float_type = {"np_float": np.float16, "torch_float": torch.float16} if use_half else {"np_float": np.float32, "torch_float": torch.float32}

        if self.transform is None:
            self.transform = lambda x: x
            
        self.mix_up = mix_up
        
    def load_data(self, idx):
        data_loc = os.path.join(self.data_dir, self.names[idx])
        label_loc = os.path.join(self.label_dir, self.names[idx])

        data_value = np.fromfile(data_loc, dtype = np.uint8).astype(self.float_type["np_float"])        # all data are stored as u8, should be converted to float
        tensor_data = torch.from_numpy(data_value).reshape(-1, self.seq_num)
        tensor_data = self.transform(tensor_data)   # preprocessing the loaded data (e.g. deterministic encoding)

        raw_label = np.fromfile(label_loc, dtype = np.uint8).astype(self.float_type["np_float"])
        tensor_label = torch.from_numpy(raw_label)
        return tensor_data, tensor_label

    def get_sequence(self, idx, output = 'atcg'):
        data_loc = os.path.join(self.data_dir, self.names[idx])
        data_value = np.fromfile(data_loc, dtype = np.uint8).reshape(-1, self.seq_num)
        if output in {'atcg', 'idx'}:
            idx = data_value.argmax(axis = 0)
            if output == 'idx':
                return idx                          # return ATCG idx: A = 0, T = 1, ...
            return ''.join(MAPPING[idx].tolist())   # return character ATCG sequence
        else:
            return data_value                       # return one hot
            
    def disable_mixup(self):
        self.mix_up = 0.0

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        seq_1, label_1 = self.load_data(idx)
        if self.mix_up > 1e-4 and self.is_train:
            new_idx = (idx + randint(1, self.data_num - 1)) % self.data_num     # make sure when mixing up, there will be no original sequences
            seq_2, label_2 = self.load_data(new_idx)
            beta = np.random.beta(self.mix_up, self.mix_up)
            mixed_seq = beta * seq_1 + (1. - beta) * seq_2
            mixed_label = beta * label_1 + (1. - beta) * label_2
            return mixed_seq, mixed_label
        return seq_1, label_1

    def __repr__(self):
        return f"CustomDataSet(\n\tdata_dir={self.data_dir},\n" \
            f"\tlabel_dir={self.label_dir}, use_half={self.use_half}\n" \
            f"is_train={self.is_train}, len={len(self.names)}\n)"

    def __str__(self):
        return self.__repr__()

    def debug_get(self, idx: int):
        dats, lbls = self.__getitem__(idx)
        print(dats.shape, dats.dtype, dats.device, lbls.shape, lbls.dtype, lbls.device)
        
if __name__ == "__main__":
    print("Test dataset integrity.")
    dataset = CustomDataSet("../data/", 500, use_half = True)
    print(dataset)
    dataset.debug_get(0)