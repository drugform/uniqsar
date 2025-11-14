import pandas as pd
import numpy as np

import sys
sys.path.append('..')
from utils import train_test_split

input_file = "selectivity_src.csv"
df = pd.read_csv(input_file)

old_cols = [f"n_offtarget_{x}" for x in [70, 75, 80, 85]]
new_cols = [f"selective_{x}" for x in [70, 75, 80, 85]]
thrs = [195, 49, 8, 1] # mean value for each column

for old_c, new_c, thr in zip(old_cols, new_cols, thrs):
    df[new_c] = (df[old_c] <= thr).astype(int)
    
# получаем веса примеров
vals = 1-df[new_cols].to_numpy()
col_ws = vals.mean(axis=0)
weights = np.ones_like(vals)
for i in range(len(col_ws)):
    col = weights[:,i]
    pos_ids = vals[:,i]==1
    col[pos_ids] = 1/col_ws[i]
    weights[:,i] = col

ws = weights.mean(axis=1)
ws /= ws.mean()
df['weight'] = (ws / ws.mean()).round(2)

cols2save = ['smiles', 'weight'] + new_cols

# Фильтрация и сохранение только нужных колонок
df = df[cols2save]

output_file = "selectivity_full.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")
train_test_split(df, 'selectivity', 0.05)
