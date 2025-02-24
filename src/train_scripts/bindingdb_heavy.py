from model import Model

if __name__ == "__main__":
    device = 'cuda:0'
    task = {'model' :
            {'descr' : "Uniqsar/protnet_v1 BindingDB heavy",
             'name' : 'bindingdb_heavy_grlr',
             'tags' : ['uniqsar', 'qsar', 'gpu', 'chemformer', 'esm']},
            
            'dataset' :
            {'name' : 'bindingdb',
             'calc' : ['pKi', 'pIC50'],
             'deriv' : {},
             'param' : ['protein']},
            
            'encoder' :
            {'name' : 'chemformer',
             'variant' : 'heavy',
             'augment' : 0,
             'minpad' : 0},
            
            'net' :
            {'name' : 'protnet_v1',
             'esm_variant' : 'normal',
             'n_layers' : 6,
             'prot_kernel' : 7,
             'prot_stride' : 3,
             'out_kernel' : 5,
             'out_stride' : 2,
             'out_downscale' : 2},
            
            'train' :
            {'batch_size' : 24,
             'n_folds' : 4,
             'train_prop' : 0.9,
             'learning_rate' : 3e-4,
             'n_epochs' : 20,
             'n_best_nets' : 3,
             'tune' : True}}
             
    
    m = Model(task, device)
