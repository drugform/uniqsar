from rdkit import Chem
from base64 import b64encode, b64decode
from base58 import b58encode, b58decode
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from molvs import Standardizer
S = Standardizer()

class Mol ():
    def __init__ (self, smiles):
        self.smiles = smiles
        self.rdmol = Chem.MolFromSmiles(smiles)
        if self.rdmol is None:
            raise Exception(f'Bad smiles: {smiles}')

        self.rdmol = S.standardize(self.rdmol)

        self.binmol = b58encode(self.rdmol.ToBinary()).decode()
        self.canon_smiles = Chem.MolToSmiles(
            self.rdmol,
            canonical=True,
            isomericSmiles=True)

        self.imgurl = f"/image/mol/{self.binmol}"
        self.molmap = {'x': 0,
                       'y': 0}

    def pack (self):
        return {'smiles' : self.smiles,
                'canonSmiles' : self.canon_smiles,
                'binmol' : self.binmol,
                'imgurl' : self.imgurl,
                'molmap' : self.molmap}

    @classmethod
    def unpack (cls, pack):
        self = cls.__new__(cls)
        self.smiles = pack['smiles']
        self.canon_smiles = pack['canonSmiles']
        self.binmol = pack['binmol']
        self.rdmol = Chem.Mol(b58decode(pack['binmol']))
        self.molmap = pack['molmap']
        return self

def augment_smiles (mol, n_augs, maxlen=None):
    sm = mol.smiles
    if n_augs != 0:
        seed = 42
        augments = Chem.MolToRandomSmilesVect(
            mol.rdmol, n_augs, randomSeed=seed)
        if maxlen is not None:
            augments = [a_ for a_ in augments if len(a_) <= maxlen]
        
        if sm in augments:
            seqs = augments
        else:
            seqs = [sm] + augments
    else:
        seqs = [sm]
        
    return seqs
