from rdkit import Chem
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
        self.canon_smiles = Chem.MolToSmiles(
            self.rdmol,
            canonical=True,
            isomericSmiles=True)

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
