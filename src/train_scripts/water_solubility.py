from model import Model

if __name__ == "__main__":
    device = 'cuda:0'
    task = {'model' :
            {'descr' : "Uniqsar/transformer_v1 water solubility model",
             'name' : 'water_solubility',
             'tags' : ['uniqsar', 'qsar', 'gpu', 'chemformer', 'transformer']},
            
            'dataset' :
            {'name' : 'water_solubility',
             'calc' : ['logs'],
             'deriv' : {},
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
            {'batch_size' : 64,
             'n_folds' : 5,
             'train_prop' : 0.9,
             'learning_rate' : 3e-4,
             'n_epochs' : 100,
             'n_best_nets' : 5,
             'tune' : True}}
    
    m = Model(task, device)
