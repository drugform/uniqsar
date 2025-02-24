import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from encoders import esm

from net_utils import ConvPooler, PositionalEncoding, LayerNorm

class TEnc (nn.Module):
    def __init__ (self, embdim, protdim, n_targets,
                  n_layers,
                  prot_kernel, prot_stride,
                  out_kernel, out_stride, out_downscale):
        super(TEnc, self).__init__()
        transformer =  nn.Transformer(d_model=embdim,
                                      dim_feedforward=embdim*4,
                                      num_encoder_layers=n_layers,
                                      batch_first=True,
                                      norm_first=True)
        self.embdim = embdim
        self.n_targets = n_targets
        self.t_enc = transformer.encoder
        del(transformer)

        outdim = int(embdim // out_downscale)
        #self.prot_pooler = ConvPooler(protdim, embdim,
        #                              kernel_size=prot_kernel,
        #                              stride=prot_stride)
        self.prot_pooler = nn.Sequential(nn.Linear(protdim, embdim),
                                         nn.ReLU(),
                                         LayerNorm(embdim))

        self.out_pooler = ConvPooler(embdim, outdim,
                                     kernel_size=out_kernel,
                                     stride=out_stride)

        self.relu = nn.ReLU()
        self.ff = nn.Linear(embdim,#outdim,
                            self.n_targets)

    def forward (self, x, prot_enc):
        sep_ = torch.empty((len(x), 1, self.embdim))
        sep_.fill_(torch.nan)
        sep = sep_.to(x.device)

        x_prot = self.prot_pooler(prot_enc)
        inp = torch.hstack((x,
                            sep,
                            x_prot))
        
        attn_mask = (inp[:,:,0]==0)
        inp.nan_to_num_(0)

        enc = self.t_enc(inp, src_key_padding_mask=attn_mask)
        #enc = self.out_pooler(enc)
        enc_ff = enc.max(dim=1)[0]
        return self.ff(enc_ff)

class Net (nn.Module):
    def __init__ (self, embdim, n_targets, n_descr, esm_variant,
                  n_layers,
                  prot_kernel, prot_stride,
                  out_kernel, out_stride, out_downscale):
        super(Net, self).__init__()
        self.esm_encoder = esm.Encoder(variant=esm_variant)
        protdim = self.esm_encoder.net_params['embdim']
        self.t_enc_ = TEnc(embdim, protdim, n_targets,
                           n_layers,
                           prot_kernel, prot_stride,
                           out_kernel, out_stride, out_downscale)
        self.t_enc = nn.DataParallel(self.t_enc_)

    def forward (self, x, d):
        protein_seqs = [d_[0] for d_ in d]
        prot_enc_lst = self.esm_encoder(protein_seqs)
        prot_enc_ = self.esm_encoder.collate_fn(prot_enc_lst)
        prot_enc = prot_enc_.to(x.device)
        return self.t_enc(x, prot_enc)
