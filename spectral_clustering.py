import torch
import numpy as np
from models.simple_moco import MoCo, SimpleEncoder
from models.seq_pred import SeqPredictor
from sklearn.cluster import SpectralClustering, KMeans
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

def get_model_embeddings(path: str, is_torch: bool = False):
    model = SeqPredictor()
    model.load(path)
    # shape (2000, C)
    weights = model.classify[-1].weight.detach()
    if is_torch:
        return weights
    return weights.numpy()

# Direct clustering without further post-processing
def vanilla_spectrum_clutering(model_path: str, label_path: str):
    gt_labels   = get_gt_label(label_path)
    embeddings  = get_model_embeddings(model_path)
    
    sclt = SpectralClustering(n_clusters = 10, n_components = 10, gamma = 0.6, n_neighbors = 16, assign_labels = 'discretize', affinity = 'nearest_neighbors')
    pred_labels = sclt.fit_predict(embeddings)
    print(f"NMI: {normalized_mutual_info_score(gt_labels, pred_labels)}, AMI: {adjusted_mutual_info_score(gt_labels, pred_labels)}")
    
def moco_spectrum_clutering(model_path: str, label_path: str, moco_path: str):
    gt_labels   = get_gt_label(label_path)
    embeddings  = get_model_embeddings(model_path, True).cuda()
    
    moco_encoder = MoCo(SimpleEncoder).cuda()
    moco_encoder.load(moco_path)
    
    reembedded = moco_encoder.base_encoder(embeddings).detach().cpu().numpy()
    embeddings = embeddings.cpu().numpy()
    sclt1 = SpectralClustering(n_clusters = 10, n_components = 10, gamma = 0.6, n_neighbors = 16, assign_labels = 'discretize', affinity = 'nearest_neighbors')
    sclt2 = SpectralClustering(n_clusters = 10, n_components = 10, gamma = 0.6, n_neighbors = 16, assign_labels = 'discretize', affinity = 'nearest_neighbors')
    moco_preds = sclt1.fit_predict(reembedded)
    seqp_preds = sclt2.fit_predict(embeddings)
    print(f"MOCO: NMI: {normalized_mutual_info_score(gt_labels, moco_preds)}, AMI: {adjusted_mutual_info_score(gt_labels, moco_preds)}")
    print(f"Seq Predictor: NMI: {normalized_mutual_info_score(gt_labels, seqp_preds)}, AMI: {adjusted_mutual_info_score(gt_labels, seqp_preds)}")
    
if __name__ == "__main__":
    moco_spectrum_clutering("./model/chkpt_2_res_bb_1000.pt", "./data_project4/celltype.txt", "./model/chkpt_2_moco_1000.pt")
    
    # vanilla_spectrum_clutering("./model/chkpt_2_res_bb_1000.pt", "./data_project4/celltype.txt")


