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
from torch.utils import data
        
class CustomDataSet(data.Dataset):
    """
        root_dir: dataset root directory, should end with "/"
    """
    def __init__(self, root_dir: str, seq_num: int, transform = None, is_train = True, use_half = False):
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
        
        self.use_half = use_half
        self.float_type = {"np_float": np.float16, "torch_float": torch.float16} if use_half else {"np_float": np.float32, "torch_float": torch.float32}

        if self.transform is None:
            self.transform = lambda x: x

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        data_loc = os.path.join(self.data_dir, self.names[idx])
        label_loc = os.path.join(self.label_dir, self.names[idx])

        data_value = np.fromfile(data_loc, dtype = np.uint8).astype(self.float_type["np_float"])        # all data are stored as u8, should be converted to float
        tensor_data = torch.from_numpy(data_value).reshape(-1, self.seq_num)
        tensor_data = self.transform(tensor_data)   # preprocessing the loaded data (e.g. deterministic encoding)

        raw_label = np.fromfile(label_loc, dtype = np.uint8).astype(self.float_type["np_float"])
        tensor_label = torch.from_numpy(raw_label)
        return tensor_data, tensor_label

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