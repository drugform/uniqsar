import sys
import pandas as pd

sys.path.append("..")
from utils import train_test_split, cold_drug_split

if __name__ == "__main__":
    src_fname = 'result_cut.csv'
    df = pd.read_csv(src_fname)
    columns = {'SMILES_1' : 'smiles',
               'SMILES_2' : 'target_smiles',
               'value' : 'similarity'}
    df = df.rename(columns=columns)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_ids, test_ids = cold_drug_split(
        df, 'smiles',
        test_part=0.1)
    df.iloc[train_ids].to_csv('mol_similarity.csv', index=False)
    df.iloc[test_ids].to_csv('mol_similarity_test.csv', index=False)
