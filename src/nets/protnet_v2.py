
import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from encoders import esm

from net_utils import ConvPooler, PositionalEncoding, LayerNorm
from transformer_layers import Encoder, Decoder

class ZeroEncoder (nn.Module):
    def __init__ (self):
        super(ZeroEncoder, self).__init__()

    def forward (self, src, mask, src_key_padding_mask):
        return src

class TEncPt (nn.Module):
    def __init__ (self, embdim, protdim, n_targets, n_descr,
                  n_layers, n_prot_layers):
        super(TEncPt, self).__init__()
        self.n_targets = n_targets
        self.embdim = embdim
        self.protdim = protdim

        if n_prot_layers == 0:
            custom_encoder = ZeroEncoder()
        else:
            custom_encoder = None
        self.T = nn.Transformer(d_model=embdim,
                                dim_feedforward=4*embdim,
                                custom_encoder=custom_encoder,
                                num_encoder_layers=n_prot_layers,
                                num_decoder_layers=n_layers,
                                batch_first=True,
                                norm_first=True,
                                dropout=0.2)
        
        self.prot_mapper_lin = nn.Sequential(
            nn.Linear(self.protdim, self.embdim, bias=False),
            LayerNorm(self.embdim))
        self.prot_mapper_conv = ConvPooler(self.protdim, self.embdim,
                                           kernel_size=5, stride=3)
        self.prot_mapper = self.prot_mapper_lin
        self.relu = nn.ReLU()
        self.ff = nn.Linear(self.embdim, self.n_targets)
        
    def forward (self, x, x_prot):        
        x_prot = self.prot_mapper(x_prot)
        x_mask = (x[:,:,0] == 0)
        prot_mask = (x_prot[:,:,0] == 0)

        enc = self.T(src=x_prot,
                     tgt=x,
                     src_key_padding_mask=prot_mask,
                     tgt_key_padding_mask=x_mask)

        enc_ff = enc.max(dim=1)[0]
        return self.ff(enc_ff)

class TEnc (nn.Module):
    def __init__ (self, embdim, protdim, n_targets, n_descr,
                  n_layers, n_prot_layers):
        super(TEnc, self).__init__()
        self.n_targets = n_targets
        self.embdim = embdim
        self.protdim = protdim

        self.encoder = Encoder(n_prot_layers, 8, embdim, embdim*4, 0.2)
        self.decoder = Decoder(n_layers, 8, embdim, embdim*4, 0.2)
        
        self.prot_mapper = nn.Sequential(
            nn.Linear(self.protdim, self.embdim, bias=False),
            LayerNorm(self.embdim))

        self.ff = nn.Sequential(
            nn.Linear(self.embdim, self.n_targets))
        
    def forward (self, x, x_prot):        
        x_prot = self.prot_mapper(x_prot)
        x_mask = (x[:,:,0] == 0).unsqueeze(-2)
        prot_mask = (x_prot[:,:,0] == 0).unsqueeze(-2)

        enc_out = self.encoder(x_prot, prot_mask)
        dec_out = self.decoder(x, enc_out, prot_mask, x_mask)

        ff = dec_out.max(dim=1)[0]
        return self.ff(ff)

class Net (nn.Module):
    def __init__ (self, embdim, n_targets, n_descr,
                  n_layers, n_prot_layers,
                  esm_variant):
        super(Net, self).__init__()
        self.esm_encoder = esm.Encoder(variant=esm_variant)
        protdim = self.esm_encoder.net_params['embdim']
        self.t_enc_ = TEnc(embdim, protdim, n_targets, n_descr,
                           n_layers, n_prot_layers)
        self.t_enc = nn.DataParallel(self.t_enc_)

    def forward (self, x, d):
        protein_seqs = [d_[0] for d_ in d]
        prot_enc_lst = self.esm_encoder(protein_seqs)
        prot_enc_ = self.esm_encoder.collate_fn(prot_enc_lst)
        prot_enc = prot_enc_.to(x.device)
        return self.t_enc(x, prot_enc)
