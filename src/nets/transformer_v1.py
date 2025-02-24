import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from fairseq.modules.transformer_layer import TransformerEncoderLayerBase, TransformerDecoderLayerBase
from fairseq.models.transformer import TransformerConfig as cfg

from net_utils import ConvPooler, PositionalEncoding, LayerNorm

class Net (nn.Module):
    def __init__ (self, embdim, n_targets, n_descr, n_layers):
        super(Net, self).__init__()
        cfg.encoder.embed_dim = embdim
        cfg.encoder.ffn_embed_dim = 4*embdim
        cfg.encoder.layers = n_layers
        cfg.encoder.normalize_before = True
        cfg.encoder.activation_dropout = 0.1
        
        self.enc = nn.ModuleList(
            [TransformerEncoderLayerBase(cfg) for _ in range(n_layers)])

        self.ff = nn.Sequential(
            LayerNorm(embdim),
            nn.Linear(embdim, n_targets))

    def forward (self, x, d):
        x_mask = (x[:,:,0] == 0)
        enc_state = x.transpose(0,1)

        for layer in self.enc:
            enc_state = layer(enc_state, encoder_padding_mask=x_mask)

        ff_inp = enc_state.transpose(1,0).max(dim=1)[0]
        return self.ff(ff_inp)
