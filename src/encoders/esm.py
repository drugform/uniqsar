import os
import numpy as np
from tqdm import tqdm
import torch

from utils import batch_split

_ESM_DIR_ = "../lib/esm/"

class Encoder ():
    def __init__ (self, variant, cached=True,
                  device='cuda', batch_size=4):
        self.device = device
        self.cached = cached
        if variant == 'light':
            self.batch_size = batch_size
            self.net_params = {'embdim' : 640}
        elif variant == 'normal':
            self.batch_size = 1
            self.net_params = {'embdim' : 1280}
        else:
            raise Exception(f'Unknown variant: {variant}')
        
        self.cache = {}
        self.load_model(variant)
        
    def load_model (self, variant):
        print()
        print(f"Loading ESM encoder:")
        print(f"variant={variant}")
        print(f"cached={self.cached}")
        print(f"device={self.device}")
        print(f"batch_size={self.batch_size}")
        print()
        
        checkpoints_path = _ESM_DIR_
        if variant == 'light':
            model_name = "esm2_t30_150M_UR50D"
            self.n_layers = 30
        elif variant == 'normal':
            model_name = "esm2_t33_650M_UR50D"
            self.n_layers = 33
        else:
            raise Exception(f'Unknown variant: {variant}')

        torch.hub.set_dir(_ESM_DIR_)
        self.model, self.alphabet = \
            torch.hub.load("facebookresearch/esm:main",
                           model_name)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.to(self.device)

    def __call__ (self, seqs):
        if self.cached:
            enc_list = self.cached_call(seqs)
        else:
            enc_list = self.call(seqs)
        torch.cuda.empty_cache()
        return self.collate_fn(enc_list)

    def cached_call (self, seqs):
        calc_seqs = []
        for i,seq in enumerate(seqs):
            if self.cache.get(seq) is None:
                calc_seqs.append(seq)

        if len(calc_seqs) > 0:
            calc_ret = self.call(calc_seqs)
            for req,ret in zip(calc_seqs,calc_ret):
                self.cache[req] = ret

        encs = []
        for seq in seqs:
            encs.append(
                self.cache[seq])

        return encs
        
    def call (self, samples):
        data = []
        for i,protein_seq in enumerate(samples):
            data.append((str(i), protein_seq))
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = self.model(batch_tokens.to(self.device),
                                 repr_layers=[self.n_layers],
                                 return_contacts=False)
        token_repr = results["representations"][self.n_layers]
        #pad_mask = (batch_tokens==self.alphabet.padding_idx)
        #token_repr[pad_mask] = 0
        #return token_repr
        
        enc_list = []
        for i, tokens_len in enumerate(batch_lens):
            enc_list.append(
                token_repr[i, 1:tokens_len-1])
                
        return enc_list

    
    def collate_fn (self, encs):
        maxlen = max([e.shape[0] for e in encs])
        embdim = encs[0].shape[1]
        packed = torch.zeros((len(encs), maxlen, embdim))
        for i,e in enumerate(encs):
            e_len = e.shape[0]
            packed[i, :e_len] = e
        return packed
