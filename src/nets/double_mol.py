import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from fairseq.modules.transformer_layer import TransformerEncoderLayerBase, TransformerDecoderLayerBase
from fairseq.models.transformer import TransformerConfig as cfg

from importlib import import_module

from drugform.lib.mol import Mol
from encoders import chemformer
from net_utils import ConvPooler, PositionalEncoding, LayerNorm

class RegTransformer (nn.Module):
    def __init__ (self, embdim, n_targets, n_descr,
                  n_enc_layers, n_dec_layers):
        super(RegTransformer, self).__init__()
        cfg.encoder.embed_dim = embdim
        cfg.encoder.ffn_embed_dim = 4*embdim
        cfg.encoder.layers = n_enc_layers
        cfg.encoder.normalize_before = True
        cfg.encoder.activation_dropout = 0.1

        cfg.decoder.embed_dim = embdim
        cfg.decoder.ffn_embed_dim = 4*embdim
        cfg.decoder.layers = n_dec_layers
        cfg.decoder.normalize_before = True
        cfg.decoder.input_dim = embdim
        cfg.decoder.output_dim = embdim
        cfg.decoder.activation_dropout = 0.1

        self.enc = nn.ModuleList(
            [TransformerEncoderLayerBase(cfg) for _ in range(n_enc_layers)])
        self.dec = nn.ModuleList(
            [TransformerDecoderLayerBase(cfg) for _ in range(n_dec_layers)])

        self.ff = nn.Sequential(
            LayerNorm(embdim),
            nn.Linear(embdim, n_targets))

         
    def forward (self, x, x_tgt):
        x_tgt_mask = (x_tgt[:,:,0] == 0)
        x_mask = (x[:,:,0] == 0)
        
        enc_state = x_tgt.transpose(0,1)
        dec_state = x.transpose(0,1)

        for layer in self.enc:
            enc_state = layer(enc_state, encoder_padding_mask=x_tgt_mask)

        for layer in self.dec:
            dec_state = layer(dec_state,
                              encoder_out=enc_state,
                              encoder_padding_mask=x_tgt_mask,
                              self_attn_padding_mask=x_mask)[0]
        
        ff = dec_state.transpose(1,0).max(dim=1)[0]
        return self.ff(ff)

class Net (nn.Module):
    def __init__ (self, embdim, n_targets, n_descr,
                  n_enc_layers, n_dec_layers, target_encoder_params):
        super(Net, self).__init__()
        self.target_encoder = import_module(
            '.'.join(['encoders',
                      target_encoder_params['name']])
        ).Encoder(**target_encoder_params, cached=True)

        self.T = RegTransformer(embdim, n_targets, n_descr,
                                n_enc_layers, n_dec_layers)
        self.T = nn.DataParallel(self.T)

    def forward (self, x, d):
        tgt_mols = [Mol(d_[0]) for d_ in d]
        tgt_enc_lst = self.target_encoder(tgt_mols, verbose=False, bs=len(x))
        tgt_enc_ = self.target_encoder.collate_fn([e[0] for e in tgt_enc_lst])
        tgt_enc = tgt_enc_.to(x.device)
        return self.T(x, tgt_enc)
        
