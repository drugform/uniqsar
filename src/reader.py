import pandas as pd
import numpy as np

from mol import Mol

def read_csv (path, props, descr=[]):
    data = pd.read_csv(path)

    for name in props + descr:
        if name not in data.keys():
            raise Exception(f'Column {name} not found in data file')
    
    if 'SMILES' in data.keys():
        name = 'SMILES'
    else:
        name = 'smiles'
        
    S = np.array(data.get(name)).tolist()
    Y = np.array(data.get(props), np.float32)
    try:
        D = np.array(data.get(descr), np.float32)
    except:
        D = np.array(data.get(descr))
    if D.shape[1] == 0:
        D = None

    weight_name = 'weight'
    if weight_name in data.keys():
        W = np.array(data.get('weight'), np.float32).reshape(-1,1)
    else:
        W = None

    good_ids = []
    M = []

    for i,sm in enumerate(S):
        m = Mol(sm)
        y = Y[i]
        not_nan_ids = np.where(~np.isnan(y))[0]
        only_nans = len(not_nan_ids) == 0
        if m is not None and not only_nans:
            good_ids.append(i)
            M.append(m)

    Y = Y[good_ids]
    if D is not None:
        D = D[good_ids]
    if W is not None:
        W = W[good_ids]
    
    return M,D,W,Y, good_ids
