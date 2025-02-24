import numpy as np
import sigfig
from sklearn.metrics import balanced_accuracy_score

from utils import round_fp, round_fmt

def standard_error_regr (Yp, Y):
    n = len(Yp)
    return np.sqrt(np.sum((Y-Yp)**2)/(n-1))

def calc_ci (Yp, Y, regression, n_boot=200, q=95):
    rng = np.random.RandomState(seed=42)
    ids = np.arange(Y.shape[0])
    
    boot_lst = []
    for i in range(n_boot):
        boot_ids = rng.choice(ids, size=ids.shape[0], replace=True)
        if regression:
            se = standard_error_regr(Yp[boot_ids], Y[boot_ids])
        else:
            se = balanced_accuracy_score(Y[boot_ids], Yp[boot_ids].round())
        boot_lst.append(se)

    se_boot = np.mean(boot_lst)
    ci_lower = np.percentile(boot_lst, (100-q)/2)
    ci_upper = np.percentile(boot_lst, 100-(100-q)/2)
    ci = (ci_lower, ci_upper)
    precision = calc_precision(se_boot, ci)
    r_ci = (round_fp(ci[0], precision),
            round_fp(ci[1], precision))
    return r_ci, precision

def calc_precision (value, interval):
    uncertainty = (interval[1] - interval[0])/2
    r_val, r_unc = sigfig.round(str(value), str(uncertainty), cutoff=9, sep='tuple', warn=False) # cutoff=35 for +half digit precision
    if '.' in r_unc:
        decimal_part = r_unc[r_unc.find('.')+1: ]
        precision = len(decimal_part)
    else:
        n_trailing_zeros = len(r_unc) - len(r_unc.rstrip('0'))
        precision = -n_trailing_zeros
        
    return precision

