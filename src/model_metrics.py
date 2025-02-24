import os
from matplotlib import pyplot as plt
import shutil
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error as mse_score, roc_auc_score as roc_auc_score_, balanced_accuracy_score
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from permetrics.regression import RegressionMetric

from utils import round_fp, round_fmt
from ci import calc_ci

def roc_auc_score (y1, y2):
    try:
        return roc_auc_score_(y1, y2)
    except:
        return np.nan

def remove_nans (Yp, Y):
    ids = np.where(~np.isnan(Y))[0]
    return Yp[ids], Y[ids]

def calc_model_metrics (tests, model_path, props, regr_flags, is_test=False):
    model_metrics_path = os.path.join(model_path,
                                    'test_metrics' if is_test else 'model_metrics')
    if os.path.exists(model_metrics_path):
        shutil.rmtree(model_metrics_path)
        
    os.mkdir(model_metrics_path)

    all_props_metrics = {}
    n_folds = len(tests)
    props, tests, regr_flags = add_avg_props(props, tests, regr_flags)
    for i,prop in enumerate(props):
        regr_flag = regr_flags[i]
        prop_tests = [ remove_nans(Yp[i], Y[i]) for Yp,Y in tests]
        Yp_all = np.hstack([t[0] for t in prop_tests])
        Y_all = np.hstack([t[1] for t in prop_tests])
        metrics_all = calc_prop_metrics(Yp_all, Y_all, regr_flag)
        fold_metrics = [calc_prop_metrics(Yp, Y, regr_flag) for Yp, Y in prop_tests]
        if n_folds == 1:
            fold_metrics = []

        all_props_metrics[prop] = [metrics_all] + fold_metrics
        if regr_flag:
            draw_prop_metrics_regr(prop_tests, metrics_all, model_metrics_path, prop)
        else:
            draw_prop_metrics_cls(prop_tests, metrics_all, model_metrics_path, prop)

        if is_test:
            Yp_i,Y_i = prop_tests[0]
            write_test_values(Yp_i, Y_i, model_metrics_path, prop)
            
    table = format_metrics_table(all_props_metrics)
    write_metrics_table(table, model_metrics_path)
    return all_props_metrics

def write_test_values (Yp, Y, model_metrics_path, prop_name):
    outfile = os.path.join(model_metrics_path,
                           prop_name+'.csv')
    with open(outfile, 'w') as fp:
        fp.write('target,predicted\n')
        for y,yp in zip(Y,Yp):
            fp.write(f'{y},{yp}\n')

def add_avg_props (props, tests, regr_flags):
    cls_ids = np.where(~np.array(regr_flags))[0]
    regr_ids = np.where(np.array(regr_flags))[0]

    props = [p for p in props]
    regr_flags = [r for r in regr_flags]
    if len(regr_ids) > 0:
        props.append('avg_regr')
        regr_flags.append(1)
    if len(cls_ids) > 0:
        props.append('avg_cls')
        regr_flags.append(0)
        
    splitted_tests = []
    for Yp,Y in tests:
        Yp_cls = Yp[:,cls_ids].ravel()
        Y_cls = Y[:,cls_ids].ravel()
        Yp_regr = Yp[:,regr_ids].ravel()
        Y_regr = Y[:,regr_ids].ravel()

        Yp_split = [Yp[:,i] for i in range(Yp.shape[1])]
        Y_split = [Y[:,i] for i in range(Y.shape[1])]
        if len(regr_ids) > 0:
            Yp_split.append(Yp_regr)
            Y_split.append(Y_regr)

        if len(cls_ids) > 0:
            Yp_split.append(Yp_cls)
            Y_split.append(Y_cls)
        
        splitted_tests.append([Yp_split, Y_split])
    
    return props, splitted_tests, regr_flags


'''
def add_avg_metrics (metrics, props, regr_flags):
    # переделать все!
    n_cols = len(list(metrics.values()[0]))
    avg_regr = [{'rmse' : []
                 'rmse_ci' : [],
                 'q2' : [],
                 'precision' : []} for _ in range(n_cols)]
    avg_cls = [{'acc' : [],
                'acc_ci': [],
                'roc_auc:' [],
                'precision': [],
                'cm' : []} for _ in range(n_cols)]
    has_regr = False
    has_cls = False
    for regr,prop in zip(regr_flags,props):
        prop_metrics = metrics[prop]
        if regr:
            has_regr = True:
            for i in range(n_cols):
                avg_regr[i]['rmse'].append(
                    prop_metrics[i]['rmse'])
                avg_regr[i]['rmse_ci'].append(
                    prop_metrics[i]['rmse_ci'])
                avg_regr[i]['q2'].append(
                    prop_metrics[i]['q2'])
                avg_regr[i]['precision'].append(
                    prop_metrics[i]['precision'])
        else:
            has_cls = True
            for i in range(n_cols):
                avg_cls[i]['acc'].append(
                    prop_metrics[i]['acc'])
                avg_cls[i]['acc_ci'].append(
                    prop_metrics[i]['acc_ci'])
                avg_cls[i]['roc_auc'].append(
                    prop_metrics[i]['roc_auc'])
                avg_cls[i]['precision'].append(
                    prop_metrics[i]['precision'])
                avg_cls[i]['cm'].append(
                    prop_metrics[i]['cm'])

    for i in range(n_cols):
        avg_regr[i]['rmse'] = np.mean(avg_regr[i]['rmse'])
        avg_regr[i]['rmse_ci'] = (np.mean(avg_regr[i]['rmse'])
'''
        



def format_metrics_table (metrics):
    rows = []
    for prop, vals in metrics.items():
        n_folds = len(vals)-1
        break
    
    header = ['', 'Total'] + [f'Fold {i}' for i in range(n_folds)]
    rows.append(header)
    
    for prop, vals in metrics.items():
        if vals[0]['regr_flag']:
            rmse_row = [f'{prop} (RMSE)'] + [v['rmse_format'] for v in vals]
            q2_row = [f'{prop} (Q2)'] + [v['q2'] for v in vals]
            conf_index_row = [f'{prop} (CI)'] + [v['conf_index'] for v in vals]
            rows.append(rmse_row)
            rows.append(q2_row)
            rows.append(conf_index_row)
        else:
            acc_row = [f'{prop} (Bal Acc)'] + [v['acc_format'] for v in vals]
            roc_auc_row = [f'{prop} (ROC AUC)'] + [v['roc_auc'] for v in vals]
            rows.append(acc_row)
            rows.append(roc_auc_row)

    return rows

def write_metrics_table (table, model_metrics_path):
    metrics_file = os.path.join(model_metrics_path,
                                'metrics.csv')
    
    with open(metrics_file, 'w') as fp:
        for row in table:
            fp.write(','.join([str(r) for r in row])+'\n')

def calc_confidence_index (Y, Yp):
    return RegressionMetric(Y, Yp).confidence_index()
            
def calc_prop_metrics (Yp, Y, regr_flag):
    # вместо несимметричного дов. интервала
    # скорректировать значение среднего до симметричного?
    if regr_flag:
        rmse_ = mse_score(Y, Yp)**0.5
        q2 = round_fp(r2_score(Y, Yp), 2)
        conf_index = round_fp(calc_confidence_index(Y, Yp), 2)
        rmse_ci, prec = calc_ci(Yp, Y, regr_flag)
        rmse = round_fp(rmse_, prec)
        d1 = round_fp(rmse - rmse_ci[0], prec)
        d2 = round_fp(rmse_ci[1] - rmse, prec)
        if abs(d1-d2) < 0.25*(d1+d2)/2: # less than 25% diff
            d = max(d1, d2)
            rmse_format = f'{round_fmt(rmse, prec)} \u00B1 {round_fmt(d, prec)}'
        else:
            rmse_format = f'{round_fmt(rmse, prec)} -{round_fmt(d1, prec)}/+{round_fmt(d2, prec)}'
        
        metrics = {'regr_flag' : 1,
                   'rmse' : rmse,
                   'rmse_ci' : rmse_ci,
                   'rmse_format' : rmse_format,
                   'q2' : q2,
                   'conf_index' : conf_index,
                   'precision' : prec}
        
    else:
        roc_auc_ = roc_auc_score(Y, Yp)
        acc_ = balanced_accuracy_score(Y, Yp.round())
        acc_ci, prec = calc_ci(Yp, Y, regr_flag)
        acc = round_fp(acc_, prec)
        roc_auc = round_fp(roc_auc_, prec)
        d1 = round_fp(acc - acc_ci[0], prec)
        d2 = round_fp(acc_ci[1] - acc, prec)
        if d1==d2:
            acc_format = f'{round_fmt(acc, prec)} \u00B1 {round_fmt(d1, prec)}'
        else:
            acc_format = f'{round_fmt(acc, prec)} -{round_fmt(d1, prec)}/+{round_fmt(d2, prec)}'
            
        cm = confusion_matrix(Y, Yp.round())

        metrics = {'regr_flag' : 0,
                   'acc' : acc,
                   'acc_ci' : acc_ci,
                   'acc_format' : acc_format,
                   'roc_auc' : roc_auc,
                   'precision' : prec,
                   'cm' : cm}

    return metrics

def draw_prop_metrics_cls (tests, metrics, model_metrics_path, prop_name):
    roc_auc_label = f"ROC AUC: {metrics['roc_auc']}"
    acc_label = f"Bal Acc {metrics['acc_format']}"
    
    fig, (ax1,ax2) = plt.subplots(1,2)
    fig.suptitle(f"Model test for prop: {prop_name}")
    plt.subplots_adjust(wspace=0.3)
    
    ###
    cm = metrics['cm']
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm)# display_labels=clf.classes_
    disp1.plot(cmap='Blues', ax=ax1, colorbar=False)
    ax1.set_title(acc_label)
    
    ###
    Yp_all = np.hstack([t[0] for t in tests])
    Y_all = np.hstack([t[1] for t in tests])
    fpr, tpr, thresh = roc_curve(Y_all, Yp_all)
    ax2.plot(fpr,tpr,label=roc_auc_label)
    
    fold_metrics = []
    for fold_id, (Yp,Y) in enumerate(tests):
        fpr, tpr, thresh = roc_curve(Y, Yp)
        ax2.plot(fpr,tpr, alpha=0.25)
        
    ax2.legend()
    
    savefile = os.path.join(model_metrics_path,
                            prop_name+'.png')
    fig.savefig(savefile, dpi=200,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.close()

    
def draw_prop_metrics_regr (tests, metrics, model_metrics_path, prop_name):
    Yp_all = np.hstack([t[0] for t in tests])
    Y_all = np.hstack([t[1] for t in tests])

    m, b = np.polyfit(Y_all, Yp_all, 1)
    
    q2_label = '$\mathregular{Q^2}$: ' + str(metrics['q2'])
    rmse_label = f"RMSE: {metrics['rmse_format']}"
    conf_index_label = f"CI: {metrics['conf_index']}"
    x1 = np.min(Y_all)
    y1 = x1*m+b
    plt.axline(xy1=(x1, y1),
               slope=m,
               color='red',
               linestyle='dashed',
               linewidth=1,
               alpha=0.5,
               label='\n'.join([q2_label, rmse_label])) #, conf_index_label]))

    plt.axline(xy1=(x1, x1),
               slope=1,
               color='black',
               linestyle='dashed',
               linewidth=1,
               alpha=0.5,
               label='Perfect line')
    

    fold_metrics = []
    for fold_id, (Yp,Y) in enumerate(tests):
        plt.scatter(Y, Yp, s=1.5,
                    alpha=0.5,
                    edgecolors='none',
                    label=None)

    plt.minorticks_on()
    plt.xlabel('Target value')
    plt.ylabel('Predicted value')
        
    plt.legend(loc='lower right')
    plt.title(f"Model test for prop: {prop_name}")
    savefile = os.path.join(model_metrics_path,
                            prop_name+'.png')
    plt.savefig(savefile, dpi=200,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.close()
