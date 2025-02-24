import re
import pandas as pd
import numpy as np

from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Descriptors import MolWt
RDLogger.DisableLog('rdApp.*')

def canonize_smiles(smiles):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:
        return None

def filter_pressure(pressure):
    '''
    Функция, которая отбрасывает давления ниже 7.6 Торр (10-2 атм)
    и выше 7600 Торр (10 атм)

    input: Давление в Торр
    return: True, если давление попадает в необходимый интервал
    '''
    if 7.6 <= pressure <= 7600:
        return True
    else:
        return False

def filter_temperature(temp):
    '''
    Функция, которая отбрасывает температуры ниже и выше Цельсий
    input: Температура в Цельсиях
    return: True, если температуры попадает в необходимый интервал
    '''
    if -196 <= temp <= 700:
        return True
    else:
        return False

def smiles_atom_tokenizer(smiles):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smiles)]
    return tokens

def filter_inorganic_and_ions(smiles):
    tokens = [token.lower() for token
              in smiles_atom_tokenizer(smiles)]

    if '.' in tokens:
        return False

    if tokens.count('c') > 1:
        return True
    else:
        return False

def filter_num_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
    if 2 < num_atoms <= 100:
        return True
    else:
        return False

def choose_biggest_smiles(smiles, smiles_list):
    if '.' in smiles:
        mol_list = smiles.split('.')
        biggest_smiles = sorted({smi: len(smi) for smi
                                 in mol_list}.items(),
                                key=lambda x: x[1], reverse=True)[0][0]
    else:
        biggest_smiles = smiles
    if (biggest_smiles in smiles_list) & \
       (biggest_smiles != smiles):
        biggest_smiles = None
    return biggest_smiles

def convert_toxicity(smiles, value):
    '''
    Конвертируем из -log(mol/kg) -> mg/kg
    '''
    mol = Chem.MolFromSmiles(smiles)
    molecular_weight = MolWt(mol)
    return 10**(-value)*molecular_weight*1000

def calc_cls_weights (values):
    n_neg = len(values) - values.sum()
    w_neg = values.sum() / n_neg
    weights = (1-values) * w_neg + values
    weights /= weights.mean()
    return weights.round(3)
    
def train_test_split (df, name, test_part):
    if type(test_part) is float:
        test_part = (np.random.rand(len(df)) < test_part)
        
    df[test_part].to_csv(name+'_test.csv', index=False)
    df[~test_part].to_csv(name+'.csv', index=False)
    
    
################################################################

from collections import Counter
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_scaffold (smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold,
                                       canonical=True,
                                       isomericSmiles=False)
    return scaffold_smiles

def key_sort (keys, random_part):
    uK, uK_ids = np.unique(keys, return_inverse=True)
    uK_ids_cnt = Counter(uK_ids)
    ids = np.argsort([-uK_ids_cnt[id] for id in uK_ids])
    n_random = int(len(ids) * random_part)
    for i in range(n_random):
        frm_pos = np.random.randint(len(ids))
        to_pos = np.random.randint(len(ids))
        ids[frm_pos], ids[to_pos] = ids[to_pos], ids[frm_pos]

    return ids

def cold_drug_split (df, smiles_column,
                     test_part, random_part=0.1):
    keys = [get_scaffold(sm) for sm
            in tqdm(df[smiles_column],
                    desc='Building scaffolds for cold drug split')]
    ids = key_sort(keys, random_part)
    test_split = int(len(ids)*(1-test_part))
    train_ids = ids[:test_split]
    test_ids = ids[test_split:]
    return train_ids, test_ids

def cold_target_split (df, protein_column,
                       test_part, random_part=0.1):
    ids = key_sort(df[protein_column].to_numpy(),
                   random_part)
    test_split = int(len(ids)*(1-test_part))
    train_ids = ids[:test_split]
    test_ids = ids[test_split:]
    return train_ids, test_ids

def cold_drug_target_split (df, smiles_column, protein_column,
                            test_part, random_part=0.05):
    drug_test_ids = cold_drug_split(df, smiles_column,
                                    test_part, random_part)[1]
    target_test_ids = cold_target_split(df, protein_column,
                                        test_part, random_part)[1]
    united_test_ids = np.hstack((drug_test_ids, target_test_ids))
    uniq_test_ids = np.unique(united_test_ids)
    np.random.shuffle(uniq_test_ids)
    test_size = int(test_part*len(df))
    test_ids = uniq_test_ids[:test_size]
    all_ids = np.arange(len(df))
    train_ids = np.delete(all_ids, test_ids)
    return train_ids, test_ids


    
