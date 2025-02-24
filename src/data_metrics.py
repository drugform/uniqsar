import os
from matplotlib import pyplot as plt
import shutil
import numpy as np

def get_regression_flags (Y):
    regression = []
    Y = np.nan_to_num(Y.copy(), 0)
    for i in range(Y.shape[1]):
        regression.append(
            len((Y[:,i][Y[:,i].nonzero()]-1).nonzero()[0]) > 0)
    return regression 

def calc_data_metrics (vals, model_path, props):
    data_metrics_path = os.path.join(model_path,
                                     'data_metrics') 

    if os.path.exists(data_metrics_path):
        shutil.rmtree(data_metrics_path)
        
    os.mkdir(data_metrics_path)

    lims = []
    regr_flags = get_regression_flags(vals)
    for i,prop in enumerate(props):
        prop_vals = vals[:,i]
        if regr_flags[i]:
            prop_lims = calc_data_prop_metrics_regr(prop_vals, data_metrics_path, prop)
        else:
            prop_lims = calc_data_prop_metrics_cls(prop_vals, data_metrics_path, prop)
        
        lims.append(prop_lims)
        
    return regr_flags, np.array(lims)
            
def calc_data_prop_metrics_cls (vals, data_metrics_path, prop_name):
    nan_ids = np.isnan(vals)
    real_vals = vals[~nan_ids]
    #nan_part = (len(vals) - len(real_vals)) / len(vals)
    n_ones = np.sum(real_vals)
    n_zeros = len(real_vals) - n_ones
    n_nans = nan_ids.sum()
    if n_nans == 0:
        labels = [f'Positive ({n_ones:.0f})', f'Negative ({n_zeros:.0f})',""]
    else:
        labels=['Positive ({n_ones:.0f})', 'Negative ({n_zeros:.0f})', 'Missing ({n_nans:.0f})']
    plt.pie([n_ones, n_zeros, n_nans], labels=labels)
            

    plt.title(f"Values distribution for prop: {prop_name}")
        
    savefile = os.path.join(data_metrics_path,
                            prop_name+'.png')
    plt.savefig(savefile,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.close()

    return [0,1]
                
def calc_data_prop_metrics_regr (vals, data_metrics_path, prop_name):
    nan_ids = np.isnan(vals)
    real_vals = vals[~nan_ids]

    nan_part = (len(vals) - len(real_vals)) / len(vals)
    plt.plot([0], color='white', label=f"Total: {len(vals)}")
    plt.plot([0], color='white', label=f"Real: {len(real_vals)}")
    plt.plot([0], color='white', label=f"{nan_part*100:.1f}% NaNs")
    xmin, xmax = real_vals.min(), real_vals.max()
        
    for bin_count in [50, 200]:
        bin_width = (xmax-xmin)/bin_count
        label=f'N bins: {bin_count} (w={bin_width:.2f})'
        plt.hist(real_vals,
                 bins=bin_count,
                 label=label)

    new_y_max = int(plt.ylim()[1]*1.07) # +7% height
    plt.ylim(0, new_y_max)
    
    font_pad = (real_vals.max() - real_vals.min()) / 200

    q = 0.005 # 99% quantile 
    q_lims = [q, 0.5, 1-q]
    q_vals = np.quantile(real_vals, q_lims, interpolation='nearest')
        
    for q_lim, q_val in zip(q_lims,
                            np.quantile(real_vals, q_lims)):
        color = 'g' if q_lim == 0.5 else 'r'
        if q_lim == q_lims[-1]:
            label = f'{(1-2*q)*100:.0f}% quantile'
        elif q_lim == q_lims[1]:
            label = 'Median'
        else:
            label = None
                
        plt.axvline(q_val,
                    color=color,
                    linestyle='dashed',
                    linewidth=1,
                    label=label),
                    #alpha=0.5)
            
        plt.text(q_val-font_pad, plt.ylim()[1], f'{q_val:.1f}',
                 horizontalalignment='right',
                 verticalalignment='top',
                 rotation=90)


    plt.legend(loc='center right', fontsize='xx-small')
    step = 1.0
    xmin = int(min(real_vals))-0.5
    xmax = int(max(real_vals))+1.5
    plt.xlim(xmin, xmax)
    plt.xticks(np.arange(np.floor(min(real_vals)),
                         np.ceil(max(real_vals)),
                         step))
    
    plt.minorticks_on()
    plt.title(f"Values distribution for prop: {prop_name}")
        
    savefile = os.path.join(data_metrics_path,
                            prop_name+'.png')
    plt.savefig(savefile, dpi=200,
                bbox_inches='tight',
                pad_inches=0.2)
    plt.close()

    return [q_vals[0], q_vals[2]]

