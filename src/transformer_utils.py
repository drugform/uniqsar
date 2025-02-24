import numpy as np
import os
import torch

def prob_score (prob):
    return np.exp(np.mean(prob))

def pad_pack (batch):
    pad_code = 0 # lm.vocab['<pad>'] must be 0
    maxlen = max([len(enc) for enc in batch])
    pack = []
    for enc in batch:
        padsize = maxlen - len(enc)
        pads = [pad_code] * padsize
        pack.append(enc+pads)

    return torch.LongTensor(pack)

def subsequent_mask (tgt):
    tgt_mask = (tgt != 0).unsqueeze(-2)
    size = tgt_mask.size(-1)
    return tgt_mask.to(torch.uint8) & torch.tril(torch.ones(1,size,size, dtype=torch.uint8)).to(tgt_mask.device)

def pack_batch (src,tgt,ws):
    with torch.no_grad():
        tgt,tgt_y = tgt[:,:-1], tgt[:,1:]
        tgt_mask = subsequent_mask(tgt)
        ntokens = (tgt_y != 0).data.sum()
        nseqs = torch.LongTensor([len(src)]).sum()
        ws = torch.FloatTensor(ws)
    return src,tgt,tgt_y,tgt_mask,ntokens,nseqs,ws

def pad_after_eos (pred):
    #pred = pred.clone()
    ends = torch.nonzero(pred==3)
    for e in ends:
        pred[e[0], e[1]+1:] = 0
    return pred

def pred_accuracy (pred, tgt_y, ntokens):
    n_pads = (tgt_y == 0).sum()
    #n_pads = int(tgt_y.view(-1).size()[0])-int(ntokens)
    pred = pad_after_eos(pred)
    n_correct_tokens = (pred == tgt_y).sum() - n_pads
    n_correct_seqs = ((tgt_y != pred).sum(dim=1) == 0).sum()
    return float(n_correct_tokens), float(n_correct_seqs)


