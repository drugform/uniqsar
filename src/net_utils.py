import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

class DataParallelWrapper (nn.DataParallel):
    def __getattr__ (self, name):
        try:
            return super()._getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class ConvPooler (nn.Module):
    def __init__ (self, indim, outdim, kernel_size, stride):
        super(ConvPooler, self).__init__()
        self.conv = nn.Conv1d(indim, outdim, kernel_size, stride)
        self.relu = nn.ReLU()
        self.norm = LayerNorm(outdim)

    def forward (self, x):
        xt = x.transpose(1,2)
        xc = self.conv(xt)
        xc = self.relu(xc)
        y = xc.transpose(1,2)
        y = self.norm(y)
        return y

class PositionalEncoding(nn.Module):
    def __init__(self, model_depth, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, model_depth)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, model_depth, 2) *
                             -(math.log(10000.0) / model_depth))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
