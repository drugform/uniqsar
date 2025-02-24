from model import Model

prime_numbers = [1,2,3,5,7,11,13,17,19,23,29,31,37]
size_filters = prime_numbers[:7]
n_filters = [64]*len(size_filters)
filters = list(zip(n_filters, size_filters))
minpad=max(size_filters)

if __name__ == "__main__":
    device = 'cuda:0'
    task = {'model' :
            {'descr' : "Uniqsar/transformer_v1 molnet_clintox benchmark model",
             'name' : 'molnet_clintox',
             'tags' : ['uniqsar', 'qsar', 'gpu', 'chemformer']},
            
            'dataset' :
            {'name' : 'molnet_clintox',
             'calc' : ['fda_approved', 'clintox_trial'],
             'deriv' : {},
             'param' : []},
            
            'encoder' :
            {'name' : 'chemformer',
             'variant' : 'light',
             'augment' : 10,
             'minpad' : minpad},
            
            'net' :
            {'name' : 'textcnn',
             'filters' : filters,
             'mlp_dim' : 64,
             'dropout' : 0.25,
             'activation' : 'swish'},
            
            'train' :
            {'batch_size' : 64,
             'n_folds' : 5,
             'train_prop' : 0.9,
             'learning_rate' : 1e-3,
             'n_epochs' : 150,
             'n_best_nets' : 5,
             'tune' : True}}
    
    m = Model(task, device)
