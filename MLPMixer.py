# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:54:27 2024

@author: 60183
"""
import torch
from torch import nn
from torch.nn import Parameter
from functools import partial



def setEmbedingModel(d_list,embed_dim):
    # for d in d_list:
    #     if d > 2000:
    #         nn.ModuleList([Mlp(d,1024,d_out)])
            
    return nn.ModuleList([Mlp(d,embed_dim)for d in d_list])

class Mlp(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.2):
        super(Mlp, self).__init__()
        # init layers
        self.fc1 = nn.Linear(in_dim, out_dim)
        
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        if self.dropout:
            out = self.dropout(out)
            
        return out

class view_PreNormResidual(nn.Module):
    def __init__(self, view_num, embed_dim, inner_dim, dropout):
        super().__init__()
        self.view_Linear = partial(nn.Conv1d, kernel_size = 1)
        self.fn = view_FeedForward(view_num, inner_dim, dropout, self.view_Linear)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        
        return self.fn(self.norm(x), mask) + x
    
class dim_PreNormResidual(nn.Module):
    def __init__(self, dim, expansion_factor, dropout):
        super().__init__()
        
        self.fn = dim_FeedForward(dim, expansion_factor, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask):
        return self.fn(self.norm(x), mask) + x

class view_FeedForward(nn.Module):
    def __init__(self, view_num, inner_dim, dropout, dense):
        super().__init__()
        self.fn1 = dense(view_num, inner_dim)
        self.fn2 = dense(inner_dim, view_num)
        self.GELU = nn.GELU()
        self.Dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        
        if mask is not None:
            mask = mask.unsqueeze(2).float()
            x = x.masked_fill(mask == 0, 0)
            
        x = self.fn1(x)
        x = self.GELU(x)
        x = self.Dropout(x)
        x = self.fn2(x)
    
        return x

class dim_FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor, dropout):
        super().__init__()
        self.inner_dim = int(dim * expansion_factor)
        self.fn1 = nn.Linear(dim, self.inner_dim)
        self.fn2 = nn.Linear(self.inner_dim, dim)
        self.GELU = nn.GELU()
        self.Dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x = self.fn1(x)
        x = self.GELU(x)
        x = self.Dropout(x)
        x = self.fn2(x)
        
        return x

class MLP_layer(nn.Module):
    def __init__(self, view_num, embed_dim, view_inner_dim, expansion_factor, dropout):
        super().__init__()
        self.view_fn = view_PreNormResidual(view_num, embed_dim, view_inner_dim, dropout)
        self.dim_fn = dim_PreNormResidual(embed_dim, expansion_factor, dropout)

    def forward(self, x, mask):
        x = self.view_fn(x, mask)
        x = self.dim_fn(x, mask)
        return x

class MLPMixer(nn.Module):
    def __init__(self, view_num=6, embed_dim=512, depth=4, view_inner_dim = 128, expansion_factor = 2, dropout = 0.):
        super().__init__()
        self.mlplayer = MLP_layer(view_num, embed_dim, view_inner_dim, expansion_factor, dropout)
        self.depth = depth
        
    def forward(self, x, mask):
        for i in range(self.depth):
            x = self.mlplayer(x, mask)
        return x

class Model(nn.Module):
    def __init__(self, input_len, d_list, num_classes, embed_dim, depth, view_inner_dim, expansion_factor, dropout, exponent):
        super().__init__()
        self.view_num = input_len
        self.embeddinglayers = setEmbedingModel(d_list,embed_dim)
        self.MLPMixer = MLPMixer(input_len, embed_dim, depth, view_inner_dim, expansion_factor,dropout)
        self.weights = Parameter(torch.softmax(torch.randn([1,self.view_num,1]),dim=1))
        self.exponent = exponent
        self.classification = nn.Linear(embed_dim, num_classes)
        self.act = nn.Sigmoid()
        self.Contrastive = nn.Linear(embed_dim, num_classes)
        
    def forward(self,x,mask=None):
        # x[v,n,d]
        # mask[bs, v]
        B = mask.shape[0]
        for i in range(self.view_num):
            x[i] = self.embeddinglayers[i](x[i])
        x = torch.stack(x,dim=1) # B,view,d
        x = self.MLPMixer(x, mask)
        x_cont = x
        x_cont = self.Contrastive(x_cont)
        
        x_weighted = torch.pow(self.weights.expand(B,-1,-1),self.exponent)
        x_weighted_mask = torch.softmax(x_weighted.masked_fill(mask.unsqueeze(2)==0, -1e9),dim=1) #[B, self.view_num, 1]
        assert torch.sum(torch.isnan(x_weighted_mask)).item() == 0
        
        x = self.classification(x)
        # x = x.mul(x_weighted)
        x = x.mul(x_weighted_mask)
        x = torch.einsum('bvd->bd',x)
        pred_x = self.act(x)
        
        return pred_x, x_cont
    
def get_model(input_len, d_list, num_classes, embed_dim=512, depth=2, view_inner_dim=64, expansion_factor=2, dropout=0., exponent=2):

    model = Model(input_len, d_list, num_classes, embed_dim, depth, view_inner_dim, expansion_factor, dropout, exponent)
    
    return model
