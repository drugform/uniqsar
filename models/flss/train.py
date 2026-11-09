from model import Model

if __name__ == "__main__":
    device = 'cuda:0'
    task = {'model' :
            {'descr' : "Uniqsar2/transformer molecular similarity model (FLSS)",
             'name' : 'flss',
             'tags' : ['uniqsar', 'qsar', 'gpu', 'chemformer', 'esm']},
            
            'dataset' :
            {'name' : 'mol_similarity',
             'calc' : ['similarity'],
             'deriv' : {},
             'param' : ['target_smiles']},
            
            'encoder' :
            {'name' : 'chemformer',
             'variant' : 'light',
             'augment' : 0,
             'minpad' : 0},
            
            'net' :
            {'name' : 'transformer_similarity',
             'n_layers' : 6,
             'mlp_dims' : [100]},
            
            'train' :
            {'batch_size' : 128,
             'n_folds' : 3,
             'train_prop' : 0.9,
             'learning_rate' : 1e-4,
             'n_epochs' : 10,
             'n_best_nets' : 5,
             'tune' : True}}
             
    
    m = Model(task, device)
