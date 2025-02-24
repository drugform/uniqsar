import numpy as np
import torch
import math
import time
from tqdm.autonotebook import tqdm
import copy
import os

from torch.optim import RAdam, Adam, SGD, NAdam

import sys
def fprint (*args):
    print(*args)
    sys.stdout.flush()

def get_regression_flags (Y):
    regression = []
    Y = np.nan_to_num(Y.copy(), 0)
    for i in range(Y.shape[1]):
        regression.append(
            len((Y[:,i][Y[:,i].nonzero()]-1).nonzero()[0]) > 0)
    return regression 


def multiclass_loss (output, target, weights=None):
    nans = torch.isnan(target)
    target = target.clone()
    output = output.clone()
    target[nans] = 0
    output[nans] = 0

    #output = torch.sigmoid(output)
    res = cross_entropy_loss(output, target)
    res[nans] = 0
    if weights is not None:
        res *= weights
    
    res = res.sum()/len(target)
    return res

def mse_loss (output, target, weights=None):
    target = target.clone()
    target[torch.isnan(target)] = output[torch.isnan(target)]
    if weights is not None:
        target = target * weights
        output = output * weights
    return torch.nn.functional.mse_loss(output, target)

cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

def mixed_weight (weights, i):
    if weights is None:
        return None

    if len(weights.shape) == 1: # class weights:
        return weights[i]

    if weights.shape[1] == 1: # batch weights only:
        return weights
    else: # batch + class weights
        return weights[:,i:i+1]

def mixed_loss (output, target, regression, weights=None):
    losses = []
    if len(regression) != len(output[0]):
        raise Exception('Invalid output data shape #1')
    for i in range(len(regression)):
        w = mixed_weight(weights, i)
        
        if regression[i] == 1:
            losses.append(mse_loss(output[:,i:i+1], target[:,i:i+1], w))
        else:
            losses.append(multiclass_loss(output[:,i:i+1], target[:,i:i+1], w))
    return torch.mean(torch.stack(losses))   

from sklearn.metrics import roc_auc_score

def roc_auc_loss (pred, target, class_weights=None):
    losses = []
    
    for i in range(len(pred[0])):
        loss = roc_auc_score(target[:,i:i+1],
                             pred[:,i:i+1])
        if class_weights:
            loss *= class_weights[i]
            
        losses.append(loss)
        
    if class_weights:
        return np.sum(losses)/np.sum(np.array(class_weights))
    else:
        return np.mean(losses)

def dataset_attr (ds, name, ignore_missing=False):
    try: return ds.__getattribute__(name)
    except:
        try: return ds.dataset.__getattribute__(name)
        except:
            if ignore_missing: return None
            else: raise Exception()

class MultilabelCriterion ():
    def __init__ (self, dataset):
        class_weights = dataset_attr(dataset, 'class_weights', ignore_missing=True)
        self.loss = torch.nn.CrossEntropyLoss(weight=class_weights)

    def __call__ (self, output, target, weights=None, unscaled=False):
        #output = torch.nn.functional.relu(output)
        return self.loss(output, target)
    
    def postproc (self, pred):
        #pred = torch.nn.functional.relu(pred)
        return torch.softmax(pred, dim=1)
        
        
class MixedCriterion ():
    def __init__ (self, dataset):
        try:
            self.regression_flags = dataset_attr(dataset, 'regression_flags')
        except:
            fprint("MESSAGE: Guessing regression flags from dataset...")
            Y = []
            ids = np.arange(len(dataset))
            np.random.shuffle(ids)
            for i in tqdm(ids[:1000]):
                item = dataset[i]
                Y.append([float(e) for e in item[-1]])

            self.regression_flags = get_regression_flags(np.array(Y))
            fprint(self.regression_flags)
            
        self.regression_flags = np.array(self.regression_flags)

        try:
            self.meanY, self.stdY = dataset_attr(dataset, 'scale')
        except:
            if self.regression_flags.max() == 0:
                fprint("MESSAGE: Classification only dataset, no scaling required...")
                self.meanY = np.zeros(len(self.regression_flags))
                self.stdY = np.ones(len(self.regression_flags))
            else:
                fprint("MESSAGE: Guessing scaling constants from dataset...")
                Y = []
                ids = np.arange(len(dataset))
                rng = np.random.default_rng(42)
                rng.shuffle(ids)
                for i in tqdm(ids[:1000]):
                    item = dataset[i]
                    Y.append([float(e) for e in item[-1]])

                self.fit_scale(Y)
                fprint(f"MESSAGE: Got constants: {self.meanY}, {self.stdY}")

    def __call__ (self, output, target, weights=None, unscaled=False):
        if unscaled:
            output = self.unscale(output)
        else:
            target = self.scale(target)
        return mixed_loss(output, target, self.regression_flags, weights)
    
    def postproc (self, pred):
        for i,r in enumerate(self.regression_flags):
            if r==0:
                pred[:,i] = torch.sigmoid(pred[:,i])
        
        pred = self.unscale(pred)
        return pred

    def fit_scale (self, Y):
        self.meanY = np.nanmean(Y, axis=0) * self.regression_flags
        self.meanY[np.isnan(self.meanY)] = 0
        Y = Y-self.meanY
        self.stdY = np.nanstd(Y, axis=0) * self.regression_flags + (1-self.regression_flags)
        
        self.stdY[np.isnan(self.stdY)] = 1
        self.stdY[self.stdY==0] = 1
        Y = Y/self.stdY
        return Y

    def scale (self, Y):
        Y = Y-to_device(self.meanY, Y.device)
        Y = Y/to_device(self.stdY, Y.device)
        return Y

    def unscale (self, Y):
        Y = Y*to_device(self.stdY, Y.device)
        Y = Y+to_device(self.meanY, Y.device)
        return Y

    
def to_device (arr, device):
    return torch.FloatTensor(arr).to(device)

'''
class Criterion ():
    def __init__ (self, loss, postproc=None):
        self.loss = loss
    
    def __call__ (self, *args, **kwargs):
        return self.loss(*args, **kwargs)
    
    def postproc (self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        else:
            return args
'''
    

################################################################

class Progress ():
    def __init__ (self, iterator, verbose):
        self.verbose = verbose
        self.iterator = tqdm(iterator) if verbose else iterator

    def __iter__ (self):
        return iter(self.iterator)

    def update (self, value):
        if self.verbose:
            if type(value) is float:
                value = round(value, 3)
            self.iterator.set_description(str(value))

    def manual (self, step):
        if self.verbose:
            self.iterator.update(step)
            
    def finish (self):
        if self.verbose:
            self.manual( len(self.iterator) - self.iterator.n )

def set_seed (seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True        
        
class PytorchTrainer ():
    def __init__ (self, net_builder, net_build_args, device='cuda', device_str=None):
        self.device = device
        self.net_builder = net_builder
        self.net_build_args = net_build_args
        self.net = net_builder(**net_build_args)
        self.net.to(self.device)
        self.device_str = device_str or device

    def _average_nets (self, save_path):
        if len(self.best_nets) > 1:
            best_score = min([b[0] for b in self.best_nets])
            self.best_nets = [b for b in self.best_nets if b[0]/best_score < 1.1]
            cur_params = dict(self.net.named_parameters())
            load_keys = self.best_nets[0][1].keys()
            for k in load_keys:
                arr = self.best_nets[0][1][k].data
                for i in range(len(self.best_nets)-1):
                    arr += self.best_nets[i+1][1][k].data
                cur_params[k].data = arr/len(self.best_nets)

            self.save(save_path)

    def save (self, path):
        net = self.net
        self.net = None
        best_nets = self.best_nets
        self.best_nets = []
        torch.save({'state_dict' : net.state_dict(),
                    'model' : self},
                   path)
        self.best_nets = best_nets
        self.net = net
        return path

    @classmethod
    def load (self, name, device='cpu', device_str=None):
        model_file = name+'.pt'
        dct = torch.load(model_file, map_location='cpu')
        model = dct['model']
        model.net = model.net_builder(**model.net_build_args)
        model.net.load_state_dict(dct['state_dict'])
        model.device = device
        model.device_str = device_str or device
        model.net.to(device)
        return model

    def set_device (self, device):
        self.device = device
        self.net.to(device)

    def parse_batch (self, batch):
        if len(batch) == 2:
            inpt, tgt = batch
            w = None
        elif len(batch) == 3:
            inpt, w, tgt = batch
            w = torch.FloatTensor(w).to(self.device)
        else:
            raise Exception('Unknown data format')

        if type(inpt) is torch.Tensor:
            inpt = [inpt]

        ### TODO allow user-defined input-to-tensor converter
        for i in range(len(inpt)):
            if type(inpt[i]) is torch.Tensor:
                inpt[i] = inpt[i].to(self.device)
            elif type(inpt[i]) is list or \
                 type(inpt[i]) is tuple:
                if len(inpt[i]) > 0:
                    if type(inpt[i][0]) is torch.Tensor:
                        inpt[i] = [inp.to(self.device) for inp in inpt[i]]
                
        
        #inpt = [b.to(self.device) for b in inpt]
        tgt = tgt.to(self.device)

        return inpt, w, tgt
        
    def train (self, dataset, name="pytorch_model",
               batch_size=16, criterion='auto', n_epochs=50, train_prop=0.9,
               learning_rate=1e-3, n_workers=0, verbose=True, epoch_hook=None,
               repeat_train=1, repeat_valid=1, n_best_nets=1, with_restarts=True,
               shuffle=True, start_epoch_num=0):
        
        self.bs = batch_size
        self.best_nets = []

        save_path     = f"{name}.pt"
        save_path_avg = f"{name}_avg.pt"

        train_split = int(len(dataset)*train_prop)

        if shuffle:
            train_ds, valid_ds = torch.utils.data.random_split(
                dataset, (train_split, len(dataset)-train_split),
                generator=torch.Generator().manual_seed(42))
        else:
            ids = np.arange(len(dataset))
            train_ds = torch.utils.data.Subset(dataset, ids[ :train_split])
            valid_ds = torch.utils.data.Subset(dataset, ids[train_split: ])
            
        collate_fn = dataset_attr(dataset, 'collate_fn', ignore_missing=True)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.bs, shuffle=True, num_workers=n_workers, collate_fn=collate_fn)
        valid_loader = torch.utils.data.DataLoader(
            valid_ds, batch_size=self.bs, shuffle=False, num_workers=n_workers, collate_fn=collate_fn)
        
        if criterion == 'auto':
            self.criterion = MixedCriterion(dataset)
        elif criterion == 'multilabel':
            self.criterion = MultilabelCriterion(dataset)
        else:
            self.criterion = criterion

            
        def init_optimizer (net=self.net):
            optimizer = RAdam(net.parameters(), learning_rate)#, weight_decay=0)
            #optimizer = Lookahead(optimizer, alpha=0.6 , k=10)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=2./(1+math.sqrt(5)),
                patience=3, min_lr=learning_rate/10)
            return optimizer, scheduler

        optimizer, scheduler = init_optimizer()
        self.train_step = 0

        def set_rate (rate):
            for i,p in enumerate(optimizer.param_groups):
                p['lr'] = rate

        def get_rate ():
            return [p['lr'] for p in optimizer.param_groups][0]

        early_thr = 0.995
        early_ratio = 0.2
        best_valid_loss = np.inf
        early_stop_counter = 0
        early_stopped = False
        restart_done = False
        start_moment = time.time()
        warmup_limit = 1000

        for ep in range(n_epochs):
            if early_stopped:
                break

            if os.path.exists(f'{name}_stop') or os.path.exists('stop'):
                try: os.remove(f'{name}_stop')
                except: pass
                try: os.remove('stop')
                except: pass
                fprint("Stopped on demand")
                break

            train_loss, valid_loss = 0,0
            train_counter, valid_counter = 0,0
            self.net.train()
            for _ in range(repeat_train):
                t = Progress(train_loader, verbose)
                for batch in t:
                    try: dataset_attr(dataset, 'switch_state')()
                    except: pass

                    if self.train_step < warmup_limit:
                        warmup_coef = (self.train_step+1)/warmup_limit
                        r = learning_rate*warmup_coef
                        set_rate(r)
                        self.train_step += 1
                    
                    inpt,w,y = self.parse_batch(batch)
                    optimizer.zero_grad()
                    yp = self.net(*inpt)
                    loss = self.criterion(yp, y, weights=w)
                    loss.backward()
                    optimizer.step()

                    report_loss = float(self.criterion(yp, y, weights=w, unscaled=True))
                    t.update(round(np.sqrt(report_loss), 5))
                    train_loss += report_loss*len(y)
                    train_counter += len(y)
            
            self.net.eval()
            with torch.no_grad():
                for _ in range(repeat_valid):
                    t = Progress(valid_loader, verbose)
                    for i,batch in enumerate(t):
                        try: dataset_attr(dataset, 'switch_state')()
                        except: pass
                        
                        inpt,w,y = self.parse_batch(batch)
                        yp = self.net(*inpt)

                        report_loss = float(self.criterion(yp, y, weights=w, unscaled=True))
                        valid_loss += report_loss*len(y)
                        valid_counter += len(y)
                        t.update(round(np.sqrt(report_loss), 5))

            train_loss = np.sqrt(train_loss/train_counter)
            valid_loss = np.sqrt(valid_loss/valid_counter)
            
            if np.isnan(train_loss) or np.isnan(valid_loss):
                fprint('Got nan and stopped')
                break

            if epoch_hook is not None:
                epoch_hook(self)

            ################################################################

            best_scores = [n[0] for n in self.best_nets]
            if len(best_scores) < n_best_nets:
                self.best_nets.append(
                    [valid_loss, copy.deepcopy(dict(self.net.named_parameters()))])
            else:
                worst_of_best_id = np.argmax(best_scores)
                if valid_loss < self.best_nets[worst_of_best_id][0]:
                    self.best_nets[worst_of_best_id] = \
                        [valid_loss, copy.deepcopy(dict(self.net.named_parameters()))]

            ################################################################
            scheduler.step(valid_loss)
            if valid_loss < early_thr*best_valid_loss:
                status_msg = "Got improvement"
                best_valid_loss = valid_loss
                early_stop_counter = 0
                restart_done = False
                self.save(save_path)
                                
            else:
                status_msg = ""

                if with_restarts and not restart_done:
                    if early_stop_counter >= 0.2*early_ratio*n_epochs:
                        optimizer, scheduler = init_optimizer()
                        restart_done = True
                        status_msg = "Restarted"

                if early_stop_counter > early_ratio*n_epochs: # early stop criterion
                    early_stopped = True
                    status_msg = "Early stopped at epoch {}".format(ep+1)
                else:
                    if status_msg == "":
                        status_msg = "{} epochs till early stop".format( int(early_ratio*n_epochs - early_stop_counter))
                early_stop_counter += 1

            fprint("MESSAGE: name: {} / train score: {} / validation score: {} / at epoch: {} / {} / elapsed time {}s / lr: {} / {}".format(
                    name, "%.4g" % train_loss, "%.4g" % valid_loss, ep+1+start_epoch_num, self.device_str, int(time.time()-start_moment), round(get_rate(), 5), status_msg))

        self._average_nets(save_path_avg)

    def predict (self, *data):
        self.net.eval()
        n = len(data[0])

        Yp = []
        with torch.no_grad():
            for i in range(math.ceil(n/self.bs)):
                data = [x.to(self.device) for x in data]
                yp = self.net(*data)
                yp = self.criterion.postproc(yp)
                yp = yp.detach().cpu().numpy()
                Yp.append(yp)

        return np.vstack(Yp)
        
    def predict_dataset (self, dataset, batch_size=1, verbose=True):
        self.net.eval()
        bs = batch_size

        collate_fn = dataset_attr(dataset, 'collate_fn', ignore_missing=True)
        
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=bs, shuffle=False, collate_fn=collate_fn)

        pred = []
        target = []
        
        with torch.no_grad():
            t = Progress(loader, verbose)
            for i, batch in enumerate(t):
                try: dataset_attr(dataset, 'switch_state')()
                except: pass
                    
                inpt,w,y = self.parse_batch(batch)
                yp = self.net(*inpt)
                yp = self.criterion.postproc(yp)
                yp = yp.detach().cpu().numpy()
                pred.append(yp)
                target.append(y.detach().cpu().numpy())

        pred = np.vstack(pred)
        target = np.vstack(target)
        torch.cuda.empty_cache()
        return pred
                
                

        
        

