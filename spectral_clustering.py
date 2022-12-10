import torch
import numpy as np
from models.seq_pred import SeqPredictor
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score

LABEL_TABLE = { 'CLP': 0, 'CMP': 1, 'GMP': 2, 'HSC': 3, 'LMPP': 4, 
                'MEP': 5, 'MPP': 6, 'pDC': 7, 'UNK': 8, 'mono': 9}

def get_gt_label(path: str):
    with open(path, 'r') as file:
        lines = file.readlines()
        result = []
        for line in lines:
            line = line[:-1]
            if not line in LABEL_TABLE:
                raise ValueError(f'Cell label {line} not found in label LUT.')
            result.append(LABEL_TABLE[line])
        return np.array(result)

def get_model_encoding(path: str):
    model = SeqPredictor()
    model.load(path)


