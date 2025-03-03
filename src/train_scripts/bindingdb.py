from model import Model

if __name__ == "__main__":
    task = {'model' :
            {'descr' : "Uniqsar/protnet BindingDB affinity model (Ki+IC50, fairseq, updated dataset)",
             'name' : 'bindingdb',
             'tags' : ['uniqsar', 'qsar', 'gpu', 'chemformer', 'esm']},
            
            'dataset' :
            {'name' : 'bindingdb',
             'calc' : ['pKi', 'pIC50'],
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
            {'batch_size' : 64,
             'n_folds' : 4,
             'train_prop' : 0.9,
             'learning_rate' : 8e-5,
             'n_epochs' : 60,
             'n_best_nets' : 5,
             'tune' : True}}
             
    
    m = Model(task)
