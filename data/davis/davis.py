import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings('ignore')
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from tdc.multi_pred import DTI

def load_dataset ():
    data = DTI(name = 'DAVIS')
    split = data.get_split('random')
    df_train = pd.concat((split['train'], split['valid']))
    df_test = split['test']
    shutil.rmtree('data')
    return df_train, df_test

def process_df (df):
    df_ = df[['Drug', 'Target', 'Y']]
    df_ = df_.rename(columns={'Drug' : 'smiles',
                              'Target' : 'protein',
                              'Y' : 'Ki'})
    df_ = filter_convert_smiles(df_)
    df_['pKi'] = df_['Ki'].apply(convert2log)
    return df_

"""
def read_source ():
    if not os.path.exists('davis_src.tsv'):
        download_source()

    columns={'X1' : 'smiles',
             'X2' : 'protein',
             'Y' : 'Ki'}
    df = pd.read_csv('davis_src.tsv',
                     sep='\t',
                     usecols=list(columns.keys()))
    df = df.rename(columns=columns)
    return df
"""

def filter_df (name, df, rm_ids):
    begin_len = len(df)
    df_filtered = df[~rm_ids]
    end_len = len(df_filtered)
    n_removed = begin_len-end_len
    perc_removed = (n_removed/begin_len*100)
    print(f'{name} filter: {n_removed} of {begin_len} ({perc_removed:.3f}%) records removed')
    df_filtered.reset_index(drop=True, inplace=True)
    return df_filtered

def filter_convert_smiles (df):
    def convert_fn (sm):
        try:
            can_sm = Chem.MolToSmiles(
                Chem.MolFromSmiles(sm),
                canonical=True,
                isomericSmiles=True)
            if can_sm is None:
                raise Exception
            if len(can_sm) > 300: # 512 limit in chemformer - handicap for augment
                raise Exception
        except:
            #print(f'bad smiles: {sm}')
            can_sm = np.nan
        
        return can_sm
    
    df['smiles'] = df['smiles'].progress_apply(convert_fn)
    
    rm_ids = df['smiles'].isna()
    return filter_df('SMILES converter', df, rm_ids)

def convert2log (val):
    return 9-np.log10(val)

if __name__ == '__main__':
    df_train, df_test = load_dataset()
    process_df(df_train).to_csv('davis.csv',
                                index=False,
                                columns=['smiles', 'protein', 'pKi'])

    process_df(df_test).to_csv('davis_test.csv',
                               index=False,
                               columns=['smiles', 'protein', 'pKi'])
