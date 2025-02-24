from model import Model

if __name__ == "__main__":
    device = 'cuda:0'
    task = {'model' :
            {'descr' : "Uniqsar/protnet DAVIS benchmark model (protnet_fair)",
             'name' : 'kiba',
             'tags' : ['uniqsar', 'qsar', 'gpu', 'chemformer', 'esm']},
            
            'dataset' :
            {'name' : 'kiba',
             'calc' : ['pKi'],
             'deriv' : {},
             'param' : ['protein']},
            
            'encoder' :
            {'name' : 'chemformer',
             'variant' : 'light',
             'augment' : 0,
             'minpad' : 0},
            
            'net' :
            {'name' : 'protnet_fair',
             'esm_variant' : 'light',
             'n_layers' : 6,
             'n_prot_layers' : 1,
             'prot_downscale' : None},
            
            'train' :
            {'batch_size' : 32,
             'n_folds' : 5,
             'train_prop' : 0.9,
             'learning_rate' : 3e-5,
             'n_epochs' : 75,
             'n_best_nets' : 5,
             'tune' : True}}
             
    
    m = Model(task, device)
