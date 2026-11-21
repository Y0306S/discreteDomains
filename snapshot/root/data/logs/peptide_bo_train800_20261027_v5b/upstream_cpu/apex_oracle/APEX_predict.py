import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import math
import sys
import os
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{file_dir}')
from APEX_models import AMP_model
from utils import *
pathogen_list = ['A. baumannii ATCC 19606', 'E. coli ATCC 11775', 'E. coli AIG221', 'E. coli AIG222', 'K. pneumoniae ATCC 13883', 'P. aeruginosa PA01', 'P. aeruginosa PA14', 'S. aureus ATCC 12600', 'S. aureus (ATCC BAA-1556) - MRSA', 'vancomycin-resistant E. faecalis ATCC 700802', 'vancomycin-resistant E. faecium ATCC 700221']
max_len = 52
word2idx, idx2word = make_vocab()
APEX_models = []
file_dir = os.path.dirname(os.path.abspath(__file__))
for a_model in glob.glob(f'{file_dir}/APEX_pathogen_models/APEX_*'):
    model = torch.load(a_model, map_location='cpu', weights_only=False)
    model.eval()
    APEX_models.append(model)
batch_size = 3000

def predict_APEX(seq_list):
    for ensemble_id in range(len(APEX_models)):
        AMP_model = APEX_models[ensemble_id].cpu().eval()
        data_len = len(seq_list)
        for i in range(int(math.ceil(data_len / float(batch_size)))):
            seq_batch = seq_list[i * batch_size:(i + 1) * batch_size]
            seq_rep = onehot_encoding(seq_batch, max_len, word2idx)
            X_seq = torch.LongTensor(seq_rep).cpu()
            AMP_pred_batch = AMP_model(X_seq).cpu().detach().numpy()
            AMP_pred_batch = 10 ** (6 - AMP_pred_batch)
            if i == 0:
                AMP_pred = AMP_pred_batch
            else:
                AMP_pred = np.vstack([AMP_pred, AMP_pred_batch])
        if ensemble_id == 0:
            AMP_sum = AMP_pred
        else:
            AMP_sum += AMP_pred
    AMP_pred = AMP_sum / float(len(APEX_models))
    return AMP_pred
apex_wrapper = predict_APEX
apex_best_wrapper = lambda x: np.min(apex_wrapper(x), axis=1)
apex_mean_wrapper = lambda x: np.mean(apex_wrapper(x), axis=1)
apex_worst_wrapper = lambda x: np.max(apex_wrapper(x), axis=1)
