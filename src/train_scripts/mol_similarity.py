from model import Model

if __name__ == "__main__":
    device = 'cuda:0'
    task = {'model' :
            {'descr' : "Uniqsar/transformer molecule similarity model (LS_align based)",
             'name' : 'mol_similarity_v1',
             'tags' : ['uniqsar', 'qsar', 'gpu', 'chemformer']},
            
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
            {'name' : 'double_mol',
             'n_enc_layers' : 0,
             'n_dec_layers' : 6,
             'target_encoder_params' : {'name' : 'chemformer',
                                        'variant' : 'light',
                                        'augment' : 0,
                                        'minpad' : 0}},
            
            'train' :
            {'batch_size' : 128,
             'n_folds' : 3,
             'train_prop' : 0.9,
             'learning_rate' : 1e-3,
             'n_epochs' : 20,
             'n_best_nets' : 5,
             'tune' : True}}
             
    
    m = Model(task, device)
