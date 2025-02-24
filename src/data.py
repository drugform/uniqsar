import os
import numpy as np
import torch

def collate_fn (batch, encoder):
    x = encoder.collate_fn([b[0][0] for b in batch])
    try:
        d = torch.FloatTensor(np.array([b[0][1] for b in batch]))
    except:
        d = np.array([b[0][1] for b in batch])
    w = torch.FloatTensor(np.array([b[1] for b in batch]))
    y = torch.FloatTensor(np.array([b[2] for b in batch]))
    return [x,d],w,y

class Data ():
    def __init__ (self, M, D, W, Y, encoder):
        self.encoder = encoder
        if D is None:
            D = np.zeros((len(M), 0))
        if W is None:
            W = np.ones((len(M), len(Y[0])))
        assert (len(M) == len(Y) == len(W) == len(D)), "Inconsistent input data"
        self.X = encoder(M)
        self.D = D
        self.W = W
        self.M = M
        self.Y = Y

    def shuffle (self):
        ids = np.arange(len(self.X))
        rng = np.random.default_rng(42)
        rng.shuffle(ids)
        
        self.X = [self.X[i] for i in ids]
        self.M = [self.M[i] for i in ids]
        self.D = self.D[ids]
        self.W = self.W[ids]
        self.Y = self.Y[ids]
        
        
        
class FoldView ():
    def __init__ (self, ds, fold_id, n_folds, is_test):
        if hasattr(ds, 'regression_flags'): 
            self.regression_flags = ds.regression_flags
            
        self.encoder = ds.encoder
        assert fold_id < n_folds

        all_ids = np.arange(len(ds.X))
        step = len(all_ids) / n_folds
        test_from = int(step * fold_id)
        test_to = int(step * (fold_id+1))

        test_ids = all_ids[test_from:test_to]
        train_ids = np.setdiff1d(all_ids, test_ids)

        if is_test:
            use_ids = test_ids
        else:
            use_ids = train_ids

        self.X = [ds.X[i] for i in use_ids]
        self.M = [ds.M[i] for i in use_ids]
        self.D = ds.D[use_ids]
        self.W = ds.W[use_ids]
        self.Y = ds.Y[use_ids]

class TrainView ():
    def __init__ (self, ds):
        if hasattr(ds, 'regression_flags'): 
            self.regression_flags = ds.regression_flags
        self.ds = ds
        self.encoder = ds.encoder
        self.good_ids, self.bad_ids = [],[]
        self.M,self.X,self.D,self.W,self.Y  = [],[],[],[],[]
        
        for i,x_list in enumerate(ds.X):
            if x_list is None:
                self.bad_ids.append(i)
            else:
                self.good_ids.append(i)
                self.M.append(ds.M[i])
                self.X.append(x_list)
                self.D.append(ds.D[i])
                self.W.append(ds.W[i])
                self.Y.append(ds.Y[i])
                
    def __len__ (self):
        return len(self.X)

    def __getitem__ (self, idx):
        x_list = self.X[idx]
        choice = np.random.randint(len(x_list))
        x = x_list[choice]
        return [x, self.D[idx]], self.W[idx], self.Y[idx]

    def collate_fn (self, batch):
        return collate_fn(batch, self.encoder)

class TestView ():
    def __init__ (self, ds):
        if hasattr(ds, 'regression_flags'): 
            self.regression_flags = ds.regression_flags
        self.ds = ds
        self.encoder = ds.encoder
        self.good_ids, self.bad_ids = [],[]
        self.M,self.X,self.D,self.W,self.Y  = [],[],[],[],[]
        
        flat_map, pos_map = [],[]
        for i,x_list in enumerate(ds.X):
            if x_list is None:
                self.bad_ids.append(i)
            else:
                self.good_ids.append(i)
                for j,x in enumerate(x_list):
                    self.M.append(ds.M[i])
                    self.X.append(x)
                    self.D.append(ds.D[i])
                    self.W.append(ds.W[i])
                    self.Y.append(ds.Y[i])

                flat_map += [i] * len(x_list)
                pos_map += list(range(len(x_list)))
            
        self.flat_map = np.array(flat_map)
        self.pos_map = np.array(pos_map)

    def __len__ (self):
        return len(self.X)

    def __getitem__ (self, idx):
        return [self.X[idx], self.D[idx]], self.W[idx], self.Y[idx]

    def gather_pred (self, pred):
        grp_sep = np.where(self.pos_map==0)[0]
        #grp_sep = np.unique(self.view, return_index=True)[1]
        grp_pred = np.split(pred, grp_sep)[1:]  # time consuming!
        Yp = np.array([np.mean(grp, axis=0) for grp in grp_pred])
        return Yp
        
    def insert_errors (self, pred):
        inpt_len = len(self.good_ids) + len(self.bad_ids)
        retval = np.empty((inpt_len, len(pred[0])))
        retval.fill(np.nan)
        retval[self.good_ids] = pred
        return retval

    def collate_fn (self, batch):
        return collate_fn(batch, self.encoder)

