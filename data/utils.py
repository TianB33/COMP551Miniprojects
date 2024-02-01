import os
import math

import numpy as np
from sklearn.preprocessing import normalize
import torch
import torch.nn.functional as F

def orientation(pos):
    u = normalize(X=pos[1:,:] - pos[:-1,:], norm='l2', axis=1)
    u1 = u[1:,:]
    u2 = u[:-1, :]
    b = normalize(X=u2 - u1, norm='l2', axis=1)
    n = normalize(X=np.cross(u2, u1), norm='l2', axis=1)
    o = normalize(X=np.cross(b, n), norm='l2', axis=1)
    ori = np.stack([b, n, o], axis=1)
    return np.concatenate([np.expand_dims(ori[0], 0), ori, np.expand_dims(ori[-1], 0)], axis=0)

def fmax(probs, labels):
    thresholds = np.arange(0, 1, 0.01)
    f_max = 0.0

    for threshold in thresholds:
        precision = 0.0
        recall = 0.0
        precision_cnt = 0
        recall_cnt = 0
        for idx in range(probs.shape[0]):
            prob = probs[idx]
            label = labels[idx]
            pred = (prob > threshold).astype(np.int32)
            correct_sum = np.sum(label*pred)
            pred_sum = np.sum(pred)
            label_sum = np.sum(label)
            if pred_sum > 0:
                precision += correct_sum/pred_sum
                precision_cnt += 1
            if label_sum > 0:
                recall += correct_sum/label_sum
            recall_cnt += 1
        if recall_cnt > 0:
            recall = recall / recall_cnt
        else:
            recall = 0
        if precision_cnt > 0:
            precision = precision / precision_cnt
        else:
            precision = 0
        f = (2.*precision*recall)/max(precision+recall, 1e-8)
        f_max = max(f, f_max)

    return f_max


class PretrainFeat:
    def __init__(self, pad=1024):
        self.pad = pad
        aa = "ACDEFGHIKLMNPQRSTVWYX"
        self.id_to_aa = {}
        for i in range(0, 21):
            self.id_to_aa[i] = aa[i]
        
        from transformers import AutoTokenizer
        self.ESM_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/model_zoom/transformers")
        
        from Pretrain_lightning.data.convertor import Converter
        self.data_converter = Converter()

    def get_seqs_angles(self, data):
        device = data.x.device
        seqs_list, angles_list, attn_mask_list = [], [], []
        for i in data.batch.unique():
            mask = data.batch == i
            seq = [self.id_to_aa[aa_id.item()] for aa_id in data.x[mask]]
            seq = self.ESM_tokenizer.encode("".join(seq), add_special_tokens=False)
            seq = torch.tensor(seq, device=device).reshape(-1,1)
            coord = data.pos[mask].cpu().numpy()
            L = coord.shape[0]
            
            feat_forward = self.data_converter.coord2angle(coord)
            feat_backward = self.data_converter.coord2angle(np.ascontiguousarray(coord[::-1,:]))
            
            angles = torch.cat([feat_forward, feat_backward.flip(0).roll(1,0)[:,1:]], dim=-1)
            angles = torch.nan_to_num(angles, nan=0.0, posinf=0.0, neginf=0.0)
            angles = angles.to(device)
            
            seq = F.pad(
                seq,
                (0,0,0, self.pad - seq.shape[0]),
                mode="constant",
                value=self.ESM_tokenizer.pad_token_id,
            )
            
            angles = F.pad(
                angles,
                (0, 0, 0, self.pad - angles.shape[0])
            )
            
            attn_mask = torch.zeros(size=(self.pad,), device=device)
            attn_mask[:L] = 1.0
        
            seqs_list.append(seq)
            angles_list.append(angles)
            attn_mask_list.append(attn_mask)

        seqs = torch.stack(seqs_list, dim=0)
        angles = torch.stack(angles_list, dim=0)
        attn_mask = torch.stack(attn_mask_list, dim=0)
        return seqs, angles, attn_mask
