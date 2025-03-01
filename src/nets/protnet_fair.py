
import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from fairseq.modules.transformer_layer import TransformerEncoderLayerBase, TransformerDecoderLayerBase
from fairseq.models.transformer import TransformerConfig as cfg

from encoders import esm
from net_utils import ConvPooler, PositionalEncoding, LayerNorm

class TEnc (nn.Module):
    def __init__ (self, embdim, protdim, n_targets, n_descr,
                  n_layers, n_prot_layers, prot_downscale):
        super(TEnc, self).__init__()
        cfg.encoder.embed_dim = embdim
        cfg.encoder.ffn_embed_dim = 4*embdim
        cfg.encoder.layers = n_prot_layers
        cfg.encoder.normalize_before = True
        cfg.encoder.activation_dropout = 0.1

        cfg.decoder.embed_dim = embdim
        cfg.decoder.ffn_embed_dim = 4*embdim
        cfg.decoder.layers = n_prot_layers
        cfg.decoder.normalize_before = True
        cfg.decoder.input_dim = embdim
        cfg.decoder.output_dim = embdim
        cfg.decoder.activation_dropout = 0.1

        self.enc = nn.ModuleList(
            [TransformerEncoderLayerBase(cfg) for _ in range(n_prot_layers)])
        self.dec = nn.ModuleList(
            [TransformerDecoderLayerBase(cfg) for _ in range(n_layers)])


        if prot_downscale is not None:
            self.prot_mapper = ConvPooler(protdim, embdim, *prot_downscale)
        else:
            self.prot_mapper = nn.Sequential(
                nn.Linear(protdim, embdim, bias=False),
                LayerNorm(embdim))
            
        self.ff = nn.Sequential(
            LayerNorm(embdim),
            nn.Linear(embdim, n_targets))

         
    def forward (self, x, x_prot):
        x_prot = self.prot_mapper(x_prot)
        x_prot_mask = (x_prot[:,:,0] == 0)
        x_mask = (x[:,:,0] == 0)
        
        enc_state = x_prot.transpose(0,1)
        dec_state = x.transpose(0,1)

        for layer in self.enc:
            enc_state = layer(enc_state, encoder_padding_mask=x_prot_mask)

        for layer in self.dec:
            dec_state = layer(dec_state,
                              encoder_out=enc_state,
                              encoder_padding_mask=x_prot_mask,
                              self_attn_padding_mask=x_mask)[0]
        
        ff = dec_state.transpose(1,0).max(dim=1)[0]
        return self.ff(ff)
        
class Net (nn.Module):
    def __init__ (self, embdim, n_targets, n_descr,
                  n_layers, n_prot_layers,
                  esm_variant, prot_downscale):
        super(Net, self).__init__()
        device = 'cuda' if torch.cuda.is_available else 'cpu'
        print(device, torch.cuda.is_available)
        1/0
        self.esm_encoder = esm.Encoder(device=device,
                                       variant=esm_variant,
                                       cached=True,
                                       batch_size=1)
        protdim = self.esm_encoder.net_params['embdim']
        self.t_enc_ = TEnc(embdim, protdim, n_targets, n_descr,
                           n_layers, n_prot_layers, prot_downscale)
        self.t_enc = nn.DataParallel(self.t_enc_)

    def forward (self, x, d):
        protein_seqs = [d_[0] for d_ in d]
        prot_enc_lst = self.esm_encoder(protein_seqs)
        prot_enc_ = self.esm_encoder.collate_fn(prot_enc_lst)
        prot_enc = prot_enc_.to(x.device)
        return self.t_enc(x, prot_enc)
