import numpy as np
from rdkit import Chem
import torch

from mol import Mol, augment_smiles

def smiles_atom_tokenizer (smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens

def reverse_vocab (vocab):
    return dict((v,k) for k,v in vocab.items())

class Encoder ():
    def __init__ (self, augment, minpad, **kwargs):
        self.augment = augment
        self.minpad = minpad
        ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
        ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        digits = '0123456789'
        punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        self.tokens = list(ascii_lowercase) + list(ascii_uppercase) +\
                      list(digits) + list(punctuation)

        self.vocab = {self.tokens[i]:i for i in range(len(self.tokens))}
        self.net_params = {'vocsize' : len(self.vocab)}

    def encode_seq (self, seq):
        return np.array([self.vocab[char] for char in seq])
        
    def __call__ (self, mols):
        mol_encs = []
        for mol in mols:
            mol_seqs = augment_smiles(mol, self.augment)
            mol_enc = [self.encode_seq(seq) for seq in mol_seqs]
            mol_encs.append(mol_enc)
            
        return mol_encs        
    
    def collate_fn (self, seqs):
        maxlen = max(max(map(len, seqs)),
                     self.minpad)
        packed = torch.zeros((len(seqs), maxlen)).to(torch.long)
        for i,x in enumerate(seqs):
            packed[i, :len(x)] = torch.LongTensor(x)

        packed = cut_pads(packed, minpad=self.minpad)
        return packed
            
def cut_pads (x, minpad=0):
    sums = x.sum(dim=0)
    last_valuable_col = int(sums.nonzero(as_tuple=False).flatten().max())+1
    x = x[:,:last_valuable_col]
    zeros = torch.zeros((x.shape[0], minpad)).to(torch.long).to(x.device)
    x = torch.cat([x,zeros], dim=1)
    return x
