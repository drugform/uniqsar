import numpy as np
import os
import sys
import shutil
import matplotlib.pyplot as plt
from matplotlib.patches import Patch as plot_patch
from importlib import import_module
from sklearn.metrics import r2_score, mean_squared_error as mse_score
import scipy.stats
import pandas as pd
import torch

from reader import read_csv
from data import Data, FoldView, TrainView, TestView
from pytorchtrainer.pytorch_trainer import PytorchTrainer, set_seed, MixedCriterion

from data_metrics import calc_data_metrics
from model_metrics import calc_model_metrics, calc_prop_metrics, \
    draw_prop_metrics_regr, draw_prop_metrics_cls, remove_nans

from mol import Mol

class Model ():
    def __init__ (self, task, device):
        set_seed(42)
        self.device = device
        self.task = task

        self.name = task['model']['name']
        self.calc_props = task['dataset']['calc']
        self.deriv_props = task['dataset'].get('deriv', [])
        self.param_props = task['dataset'].get('param', [])
        self.props = self.calc_props + \
                     list(self.deriv_props.keys())
        
        self.net_params = task['net']
        self.train_params = task['train']
        self.encoder_params = task['encoder']
        
        self.model_path = os.path.join("../models",
                                       self.name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        try:
            shutil.copy(sys.argv[0],
                        os.path.join(self.model_path,
                                     'train.py'))
        except shutil.SameFileError:
            pass


        self.net_builder = import_module(
            '.'.join(['nets',
                      self.net_params['name']])
        ).Net

        self.encoder = import_module(
            '.'.join(['encoders',
                      self.encoder_params['name']])
        ).Encoder(**self.encoder_params)
        # read data
        data_name = task['dataset']['name']
        data_file = os.path.join("../data/",
                             data_name,
                             data_name+".csv")
        M,D,W,Y,good_ids = read_csv(data_file,
                                    self.calc_props,
                                    self.param_props)

        self.regression_flags, self.Y_lims = \
          calc_data_metrics(Y,
                            self.model_path,
                            self.calc_props)

        Y_clip = np.clip(Y, self.Y_lims[:,0], self.Y_lims[:,1])
        dataset = Data(M,D,W,Y_clip, self.encoder)
        dataset.shuffle()
        self.net_params.update( self.encoder.net_params )
        
        self.net_params['n_targets'] = len(self.calc_props)
        self.net_params['n_descr'] = len(self.param_props)
        del(self.net_params['name'])
        self.submodel_names = []
        
        # train
        tests = []
        for fold_id in range(self.train_params['n_folds']):
            self.train_submodel(fold_id, dataset)
            tests.append(
                self.test_submodel(fold_id, dataset))

        
        self.metrics = calc_model_metrics(tests,
                                          self.model_path,
                                          self.calc_props,
                                          self.regression_flags)
        
        torch.save(self,
                   os.path.join(self.model_path,
                                'model.pt'))

        self.test()

                    
    def train_submodel (self, fold_id, dataset):
        
        submodel_name = os.path.join(self.model_path,
                                  f"fold_{fold_id}")
        self.submodel_names.append(submodel_name)
        if os.path.exists(submodel_name+'.pt'):
            print(f'Fold {fold_id} already trained, skipping')
            return

        set_seed(42)
        submodel = PytorchTrainer(self.net_builder,
                                  self.net_params,
                                  self.device)

        train_dataset = TrainView(
            FoldView(dataset,
                     fold_id,
                     self.train_params['n_folds'],
                     is_test=False))
        #print(f'Data hash: {np.nanmean(train_dataset.Y)/np.nanstd(train_dataset.Y)} (Fold {fold_id+1}, train)')
        #self.tmp.append(['train', fold_id, train_dataset])
        submodel.train(train_dataset, submodel_name,
                       batch_size=self.train_params['batch_size'],
                       criterion='auto',
                       train_prop=self.train_params['train_prop'],
                       learning_rate=self.train_params['learning_rate'],
                       n_epochs=self.train_params['n_epochs'],
                       n_workers=0,
                       n_best_nets=1,
                       with_restarts=True,
                       verbose=self.train_params.get('verbose', False),
                       shuffle=False)

        if self.train_params['tune'] == True:
            submodel =  PytorchTrainer.load(submodel_name,
                                            self.device)
            submodel.train(train_dataset, submodel_name,
                        batch_size=self.train_params['batch_size'],
                        criterion='auto',
                        train_prop=self.train_params['train_prop'],
                        learning_rate=self.train_params['learning_rate']/3,
                        n_epochs=min(self.train_params['n_epochs']//3, 30),
                        n_workers=0,
                        n_best_nets=self.train_params['n_best_nets'],
                        with_restarts=False,
                        verbose=self.train_params.get('verbose', False),
                        shuffle=True)

            if os.path.exists(submodel_name+"_avg.pt"):
                os.rename(submodel_name+"_avg.pt",
                          submodel_name+".pt")
        
        
    @classmethod
    def load (self, model_name, device):
        model_file = os.path.join("../models",
                                     model_name,
                                     "model.pt")
        obj = torch.load(model_file)
        obj.device = device
        obj.submodels = []
        for submodel_name in obj.submodel_names:
            obj.submodels.append(
                PytorchTrainer.load(submodel_name,
                                    device))
    
        obj.encoder = import_module(
            '.'.join(['encoders',
                      obj.encoder_params['name']])
        ).Encoder(**obj.encoder_params)

        return obj

    def test_submodel (self, fold_id, dataset):
        test_dataset = TestView(
            FoldView(dataset,
                     fold_id,
                     self.train_params['n_folds'],
                     is_test=True))
        bs = self.train_params['batch_size']
        submodel_name = os.path.join(self.model_path,
                                  f"fold_{fold_id}")
        submodel =  PytorchTrainer.load(submodel_name,
                                        self.device)

        pred = submodel.predict_dataset(test_dataset, batch_size=bs, verbose=False)
        Yp = test_dataset.gather_pred(pred)
        Y  = test_dataset.gather_pred(test_dataset.Y)
        return Yp,Y
    
    def predict (self, M, D=None, batch_size=None):
        if batch_size is None:
            batch_size = self.train_params['batch_size']
        Y = np.zeros((len(M), self.net_params['n_targets']))
        W = None
        
        dataset = Data(M,D,W,Y, self.encoder)
        test_dataset = TestView(dataset)

        if len(test_dataset) == 0:
            empty = np.zeros((0, self.net_params['n_targets']))
            return empty, empty


        def apply_submodel (sm_id):
            submodel = self.submodels[sm_id]

            print(f'Running submodel {sm_id}')
            pred = submodel.predict_dataset(test_dataset,
                                            batch_size=batch_size,
                                            verbose=True)
            Yp_ = test_dataset.gather_pred(pred)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return Yp_

        sub_preds = np.array(list(map(apply_submodel,
                                      np.arange(len(self.submodels)))))
        # sub_preds = np.array(async_map(apply_submodel,
        #                                np.arange(len(self.submodels))))
        Yp = np.mean(sub_preds, axis=0)
        
        ###
        stds = np.std(sub_preds, ddof=1, axis=0)
        n = len(self.submodels)
        student_coef = scipy.stats.t.ppf((1 + 0.95) / 2., n-1)
        ci_norm = student_coef * stds / np.sqrt(n)

        Yp_e = test_dataset.insert_errors(Yp)
        ci_norm_e = test_dataset.insert_errors(ci_norm)
        return Yp_e, ci_norm_e
        
    def test (self):
        data_name = self.task['dataset']['name']
        data_file = os.path.join("../data/",
                             data_name,
                             data_name+"_test.csv")
        
        if not os.path.exists(data_file):
            return

        print()
        print(f'Test file found: {data_file}')

        M,D,W,Y,good_ids = read_csv(data_file,
                                    self.calc_props,
                                    self.param_props)
        Y_clip = np.clip(Y, self.Y_lims[:,0], self.Y_lims[:,1])

        model = Model.load(self.name, self.device)
        Yp, ci = model.predict(M, D)

        self.metrics = calc_model_metrics([[Yp, Y_clip]],
                                          self.model_path,
                                          self.calc_props,
                                          self.regression_flags,
                                          is_test=True)
        
    
    def predict_file (self, input_file, output_file, batch_size=None):
        print()
        M,D,_,_,good_ids = read_csv(input_file,
                                    self.calc_props,
                                    self.param_props)
        Yp, ci = self.predict(M, D, batch_size)
        out = {'mol_id' : good_ids}
        for i,prop in enumerate(self.props):
            prec = self.metrics[prop][0]['precision']
            out[prop] = Yp[:,i].round(prec)
            
        pd.DataFrame(out).to_csv(output_file, index=False)
        print('Wrote output csv file')
        return out
        
        
