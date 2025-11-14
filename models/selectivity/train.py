from model import Model

props = {'selective_70' : (1, 0),
         'selective_75' : (1, 0),
         'selective_80' : (1, 0),
         'selective_85' : (1, 0)}

task = {'model' :
        {'descr' : "Uniqsar/transformer_v1 Target-Agnostic Selectivity model",
         'name' : 'selectivity',
         'tags' : ['uniqsar', 'qsar', 'gpu', 'chemformer', 'transformer']},
            
        'dataset' :
        {'name' : 'selectivity',
         'calc' : list(props.keys()),
         'deriv' : {'selective' : props},
         'param' : []},
            
        'encoder' :
        {'name' : 'chemformer',
         'variant' : 'light',
         'augment' : 0,
         'minpad' : 0},
            
        'net' :
        {'name' : 'transformer_v1',
         'n_layers' : 6},
            
        'train' :
        {'batch_size' : 128,
         'n_folds' : 4,
         'train_prop' : 0.9,
         'learning_rate' : 3e-4,
         'n_epochs' : 100,
         'n_best_nets' : 5,
         'tune' : True}}


if __name__ == "__main__":
    device = 'cuda:0'    
    m = Model(task, device)
