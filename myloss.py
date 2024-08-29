# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:14:16 2024

@author: 60183
"""

import torch
import sys
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


"my_contrast_loss"

class Loss(nn.Module):
    def __init__(self,alpha,device):
        super(Loss, self).__init__()
        self.device = device
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def contrastive_loss(self, x, label, inc_V_ind, inc_L_ind):
        # x = [bs, v, dim] 
        # label = [bs, c]  c = dim
        # inc_V_ind = [bs, v]
        # inc_L_ind = [bs, c]
        x = F.normalize(x, p=2, dim=-1)
        x = x.transpose(0,1)  #[v,bs,d]
        v = x.size(0)
        view_loss = 0
        sample_view_loss = 0
        label = label.mul(inc_L_ind)
        
        for i in range(v):
            for j in range(i+1, v):
                    v1, v2 = x[i,:,:], x[j,:,:]  #[bs,d]
                    mask_v1, mask_v2 = inc_V_ind[:, i], inc_V_ind[:, j]  #[bs]
                    
                    mask_both_miss = mask_v1.mul(mask_v2).bool()
                    v1, v2 = v1[mask_both_miss], v2[mask_both_miss]
                    # print('mask_both_miss', mask_both_miss.shape)
                    # print('label', label.shape)
                    label_mask = label[mask_both_miss]
                    n = v1.size(0)
                    N = 2 * n
                    
                    label_or_mun = label_mask.unsqueeze(2) + label_mask.T.unsqueeze(0)
                    label_or_mun = torch.sum(torch.where(label_or_mun > 1, 1, label_or_mun),dim=1)
                    label_matrix = (torch.matmul(label_mask, label_mask.T) / (label_or_mun + 1e-9)).fill_diagonal_(0)
                    
                    label_matrix = torch.cat([torch.cat([label_matrix, label_matrix], dim=1),
                                                  torch.cat([label_matrix, label_matrix], dim=1)], dim=0)
                    
                    label_matrix = torch.where(label_matrix == 1, 1e-5, label_matrix)
                    label_matrix = torch.where(label_matrix == 0, 1, label_matrix)
                    label_matrix = torch.where(label_matrix < 1, 0, label_matrix)
                    
                    z_v = torch.cat((v1, v2), dim=0)
                    #similarity_mat = (1+torch.matmul(z_v, z_v.T)) / 2
                    similarity_mat = (torch.matmul(z_v, z_v.T)) / self.alpha
                    similarity_mat = similarity_mat.fill_diagonal_(0)
                    similarity_mat = similarity_mat * label_matrix 
                    new_label = torch.cat((torch.tensor(range(n,N)),torch.tensor(range(0,n)))).to(self.device)  # .long()
                    view_loss += self.criterion(similarity_mat, new_label)/N

        return view_loss
