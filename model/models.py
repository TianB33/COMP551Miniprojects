import os.path as osp
from functools import partial
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max, scatter_add
import torch_geometric.transforms as T
from torch_geometric.nn import MLP, fps, global_max_pool, global_mean_pool, radius

from model.modules import *

class Linear(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 batch_norm: bool = True,
                 dropout: float = 0.0,
                 bias: bool = False,
                 leakyrelu_negative_slope: float = 0.1,
                 momentum: float = 0.2) -> nn.Module:
        super(Linear, self).__init__()

        module = []
        if batch_norm:
            module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        module.append(nn.Linear(in_channels, out_channels, bias = bias))
        self.module = nn.Sequential(*module)

    def forward(self, x):
        return self.module(x)

class MLP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 batch_norm: bool,
                 dropout: float = 0.0,
                 bias: bool = True,
                 leakyrelu_negative_slope: float = 0.2,
                 momentum: float = 0.2) -> nn.Module:
        super(MLP, self).__init__()

        module = []
        if batch_norm:
            module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        if mid_channels is None:
            module.append(nn.Linear(in_channels, out_channels, bias = bias))
        else:
            module.append(nn.Linear(in_channels, mid_channels, bias = bias))
        if batch_norm:
            if mid_channels is None:
                module.append(nn.BatchNorm1d(out_channels, momentum=momentum))
            else:
                module.append(nn.BatchNorm1d(mid_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        if mid_channels is None:
            module.append(nn.Dropout(dropout))
        else:
            module.append(nn.Linear(mid_channels, out_channels, bias = bias))

        self.module = nn.Sequential(*module)

    def forward(self, input):
        return self.module(input)

class BasicBlock(nn.Module):
    def __init__(self,
                 r: float,
                 l: float,
                 kernel_channels: list[int],
                 in_channels: int,
                 out_channels: int,
                 base_width: float = 16.0,
                 batch_norm: bool = True,
                 dropout: float = 0.0,
                 bias: bool = False,
                 leakyrelu_negative_slope: float = 0.1,
                 momentum: float = 0.2) -> nn.Module:

        super(BasicBlock, self).__init__()

        if in_channels != out_channels:
            self.identity = Linear(in_channels=in_channels,
                                  out_channels=out_channels,
                                  batch_norm=batch_norm,
                                  dropout=dropout,
                                  bias=bias,
                                  leakyrelu_negative_slope=leakyrelu_negative_slope,
                                  momentum=momentum)
        else:
            self.identity = nn.Sequential()

        width = int(out_channels * (base_width / 64.))
        self.input = MLP(in_channels=in_channels,
                         mid_channels=None,
                         out_channels=width,
                         batch_norm=batch_norm,
                         dropout=dropout,
                         bias=bias,
                         leakyrelu_negative_slope=leakyrelu_negative_slope,
                         momentum=momentum)
        self.conv = CDConv(r=r, l=l, kernel_channels=kernel_channels, in_channels=width, out_channels=width)
        self.output = Linear(in_channels=width,
                             out_channels=out_channels,
                             batch_norm=batch_norm,
                             dropout=dropout,
                             bias=bias,
                             leakyrelu_negative_slope=leakyrelu_negative_slope,
                             momentum=momentum)

    def forward(self, x, pos, seq, ori, batch):
        identity = self.identity(x)
        x = self.input(x)
        x = self.conv(x, pos, seq, ori, batch)
        out = self.output(x) + identity
        return out

class Model(nn.Module):
    def __init__(self,
                 geometric_radii: List[float],
                 sequential_kernel_size: float,
                 kernel_channels: List[int],
                 channels: List[int],
                 use_pretrain = 1,
                 base_width: float = 16.0,
                 embedding_dim: int = 1024,
                 batch_norm: bool = True,
                 dropout: float = 0.2,
                 bias: bool = False,
                 num_classes: int = 384,
                 pretrain_model_type = 'esm',
                 vq_version='stage2',
                 esm_version='ESM35M') -> nn.Module:

        super().__init__()

        assert (len(geometric_radii) == len(channels)), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"
        self.use_pretrain = use_pretrain
        self.pretrain_model_type = pretrain_model_type
        self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=embedding_dim)
        self.local_mean_pool = AvgPooling()

        layers = []
        in_channels = embedding_dim
        for i, radius in enumerate(geometric_radii):
            layers.append(BasicBlock(r = radius,
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = in_channels,
                                     out_channels = channels[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))
            layers.append(BasicBlock(r = radius,
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = channels[i],
                                     out_channels = channels[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))
            in_channels = channels[i]

        self.layers = nn.Sequential(*layers)

        self.classifier = MLP(in_channels=channels[-1],
                              mid_channels=max(channels[-1], num_classes),
                              out_channels=num_classes,
                              batch_norm=batch_norm,
                              dropout=dropout)
        
        project_dim = 0
        if "base" in pretrain_model_type:
            self.pretrain_proj_base = nn.Linear(embedding_dim,embedding_dim)
            self.pretrain_gate_base = nn.Linear(embedding_dim,1)
            # project_dim += embedding_dim
        
        if "esm" in pretrain_model_type:
            if esm_version in ['ESM35M', 'ESM35M_data1M', 'ESM35M_data1M_pad512', 'ESM35M_data1M_pad512_flash']:
                self.esm_dim=480
            if esm_version == 'ESM650M':
                self.esm_dim = 1280
            if esm_version == 'ESM3B':
                self.esm_dim = 2560
            self.pretrain_proj_esm = nn.Linear(self.esm_dim,embedding_dim)
            self.pretrain_gate_esm = nn.Linear(self.esm_dim,1)
            
        
        
        self.pretrain_proj = nn.Linear(project_dim,embedding_dim)


    def forward(self, data):
        input_feats = 0
        if 'base' in self.pretrain_model_type:
            feat = self.embedding(data.x)
            feat = self.pretrain_proj_base(feat)*F.sigmoid(self.pretrain_gate_base(feat))
            input_feats += feat
        
        cumdim = 0
        if 'esm' in self.pretrain_model_type:
            feat = data.pretrain_embedding[:, cumdim:cumdim+self.esm_dim]
            feat = self.pretrain_proj_esm(feat)*F.sigmoid(self.pretrain_gate_esm(feat))
            input_feats += feat
            cumdim += self.esm_dim
        
        
        # input_feats = torch.cat(input_feats, dim=-1)
        
        # input_feats.append(data.pretrain_embedding)
        # input_feats = torch.cat(input_feats, dim=-1)
        x, pos, seq, ori, batch = (input_feats, data.pos, data.seq, data.ori, data.batch)
            
        # if self.pretrain_model_type=='base':
        #     x, pos, seq, ori, batch = (self.embedding(data.x) , data.pos, data.seq, data.ori, data.batch)
        # elif self.pretrain_model_type in ['base+vq', 'base+esm', 'base+esm+vq']:
        #     base_embed = self.embedding(data.x)
        #     x, pos, seq, ori, batch = (self.pretrain_proj(torch.cat([base_embed, data.pretrain_embedding], dim=-1)), data.pos, data.seq, data.ori, data.batch)

        
        # x = torch.ones_like(x)

        for i, layer in enumerate(self.layers):
            x = layer(x, pos, seq, ori, batch)
            if i == len(self.layers) - 1:
                # x = torch.cat([scatter_mean(x, batch, dim=0), scatter_add(x, batch, dim=0)], dim=-1)
                # x = scatter_mean(x, batch, dim=0)
                # x = scatter_add(x, batch, dim=0)
                x = global_mean_pool(x, batch)
            elif i % 2 == 1:
                x, pos, seq, ori, batch = self.local_mean_pool(x, pos, seq, ori, batch)

        logits = self.classifier(x)

        return logits
