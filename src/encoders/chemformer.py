import sys
import os
import math
import torch
from tqdm import tqdm
from argparse import Namespace

_CF_DIR_ = "../lib/Chemformer"

def maybe_download_chemformer_code ():
    url = "https://github.com/MolecularAI/Chemformer"
    commit_hash = "a779f6e"
    path = os.path.join(_CF_DIR_, 'src')
    if os.path.exists(path):
        return

    print("No local Chemformer code, trying to download")
    os.system(f"git clone {url} {path}")
    os.system(f"git -C {path} checkout {commit_hash}")

maybe_download_chemformer_code()

sys.path.append(os.path.join(_CF_DIR_, 'src'))
from molbart.modules.data.util import BatchEncoder
from molbart.models import Chemformer
import molbart.modules.util as util
sys.path.pop()

from mol import Mol, augment_smiles
from utils import batch_split

class Encoder ():
    def __init__ (self, variant, augment, minpad,
                  device='cuda', batch_size=64,
                  name=None):
        self.augment = augment
        self.minpad = minpad
        self.device = device
        self.maxlen = 512-2
        self.batch_size = batch_size
        self.load_model(variant)

    def check_model_file (self, variant, model_path):
        source_url = "https://az.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq"
        if variant == 'light':
            src = "models/pre-trained/combined/step=1000000.ckpt"
            dest = "lib/Chemformer/models/combined.ckpt"
        elif variant == 'heavy':
            src = "models/pre-trained/combined-large/step=1000000.ckpt"
            dest = "lib/Chemformer/models/combined_large.ckpt"
            
        if not os.path.exists(model_path):
            raise Exception(f'Chemformer model file not found. You need to manually download file "{src}" from "{source_url}" in browser and put it locally as "{dest}"')  

    def fix_vocab_size_param (self, path):
        model = torch.load(path)
        model["hyper_parameters"]["vocabulary_size"] = model["hyper_parameters"]["vocab_size"]
        torch.save(model, path)
        del model
        
    def load_model (self, variant):
        print(f"Loading Chemformer encoder:")
        print(f"variant={variant}")
        print(f"device={self.device}")
        print(f"batch_size={self.batch_size}")
        print()
        args = Namespace()
        args.batch_size = self.batch_size
        args.n_gpus = 1
        args.model_type = "bart"
        args.n_beams = 10
        args.task = 'mol_opt'

        if variant == 'light':
            args.model_path = os.path.join(_CF_DIR_, 'models/combined.ckpt')
        elif variant == 'heavy':
            args.model_path = os.path.join(_CF_DIR_, 'models/combined_large.ckpt')
            self.maxlen = 512-2
        else:
            raise Exception(f'Unknown variant: {variant}. Expected one of: light, heavy')
        self.check_model_file(variant, args.model_path)
        self.fix_vocab_size_param(args.model_path)
        
        model_args, data_args = util.get_chemformer_args(args)
        
        kwargs = {
            "vocabulary_path": os.path.join(_CF_DIR_, 'src',
                                            'bart_vocab.json'),
            "n_gpus": args.n_gpus,
            "model_path": args.model_path,
            "model_args": model_args,
            "data_args": data_args,
            "n_beams": args.n_beams,
            "train_mode": "eval",
            "datamodule_type": None,
            "device" : self.device
        }

        
        self.cmf = Chemformer(**kwargs)
        del self.cmf.model.decoder
        torch.cuda.empty_cache()
        self.batch_encoder = BatchEncoder(self.cmf.tokenizer,
                                          masker=None,
                                          max_seq_len=self.maxlen)
        
    def calc (self, seqs):
        with torch.no_grad():
            inpt_, pad_mask_ = self.batch_encoder(seqs)
            inpt = inpt_.to(self.device)
            pad_mask = pad_mask_.to(self.device).T
        
            embs = self.cmf.model._construct_input(inpt)
            encs_ = self.cmf.model.encoder(
                embs, src_key_padding_mask=pad_mask)
            encs = encs_.transpose(0,1)

        enc_list = []
        for i, pads in enumerate(pad_mask):
            pad_ids = torch.where(pads)[0]
            if len(pad_ids) == 0:
                enc_item = encs[i]
            else:
                pad_from = int(pad_ids.min())
                enc_item = encs[i][:pad_from]

            enc_list.append(
                enc_item.detach().cpu())
                
        return enc_list
                
        #encs[pad_mask] = 0
        #return encs.detach()
    
    def __call__ (self, mols):
        pos_map = []
        flat_seqs = []
        flat_ids = []
        
        for i,mol in enumerate(mols):
            mol_seqs = augment_smiles(mol, self.augment, maxlen=self.maxlen)
            flat_seqs += mol_seqs
            flat_ids += [i]*len(mol_seqs)
            pos_map += list(range(len(mol_seqs)))

        
        batches = batch_split(flat_seqs, self.batch_size)
        n_batches = math.ceil(len(flat_seqs) /
                              self.batch_size)
        flat_encs = []
        for batch_seqs in tqdm(batches,
                               desc='Encoding SMILES',
                               total=n_batches):
            flat_encs += self.calc(batch_seqs)

        encs = [[] for _ in mols]
        for i,enc in zip(flat_ids, flat_encs):
            encs[i].append( enc )
        
        #grp_sep = np.where(np.array(pos_map)==0)[0]
        #encs = np.split(flat_encs, grp_sep)[1:]
        return encs

    def collate_fn (self, encs):
        maxlen = max(max([e.shape[0] for e in encs]),
                     self.minpad)
        embdim = encs[0].shape[1]
        packed = torch.zeros((len(encs), maxlen, embdim))
        for i,e in enumerate(encs):
            e_len = e.shape[0]
            packed[i, :e_len] = e

        return packed
