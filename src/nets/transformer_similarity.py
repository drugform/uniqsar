import numpy as np

import torch
import torch.nn as nn

from mol import Mol

from encoders.chemformer import Encoder as Chemformer
from net_utils import MLP

class Net (nn.Module):
    def __init__ (self, embdim, n_targets, n_descr,
                  n_layers=6, n_heads=8,
                  dropout=0.1, mlp_dims=[]):
        super(Net, self).__init__()
        self.chemformer = Chemformer(
            augment=0, minpad=0, variant='light')
        embdim = self.chemformer.net_params['embdim']

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = embdim,
            nhead = n_heads,
            dim_feedforward = embdim*4,
            dropout = dropout,
            batch_first=True)

        self.T_enc = nn.TransformerEncoder(
            encoder_layer,
            n_layers,
            norm=nn.LayerNorm(embdim),
            enable_nested_tensor=False,
            mask_check=True)

        self.ff = MLP(indim=embdim,
                      hiddims=mlp_dims+[n_targets],
                      dropout=dropout*2)

    def forward (self, x, d):
        target_mols = [Mol(d_[0]) for d_ in d]
        target_enc_lst = [enc[0] for enc in # first of the aug variant list
                           self.chemformer(target_mols)]
        target_enc_ = self.chemformer.collate_fn(target_enc_lst)
        target_enc = target_enc_.to(x.device)
        return self.forward_(x, target_enc)
        
    def forward_ (self, x, tgt):
        sep = torch.zeros_like(x[:,0:1,:])
        inp = torch.concat((x, sep, tgt), dim=1)
        x_mask = (x[:,:,0] == 0)
        tgt_mask = (tgt[:,:,0] == 0)
        sep_mask = sep[:,:,0].to(bool)
        inp_mask = torch.concat((x_mask, sep_mask, tgt_mask), dim=1)
        
        enc = self.T_enc(inp,
                         src_key_padding_mask=inp_mask)

        ff_inp = enc.max(dim=1)[0]
        return self.ff(ff_inp)

    
