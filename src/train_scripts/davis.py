from model import Model

if __name__ == "__main__":
    task = {'model' :
            {'descr' : "Uniqsar/protnet DAVIS benchmark model (protnet v1)",
             'name' : 'davis',
             'tags' : ['uniqsar', 'qsar', 'gpu', 'chemformer', 'esm']},
            
            'dataset' :
            {'name' : 'davis',
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
            {'batch_size' : 16,
             'n_folds' : 5,
             'train_prop' : 0.9,
             'learning_rate' : 3e-5,
             'n_epochs' : 75,
             'n_best_nets' : 5,
             'tune' : True}}
             
    
    m = Model(task)
