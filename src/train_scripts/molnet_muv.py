from model import Model

if __name__ == "__main__":
    device = 'cuda:0'
    task = {'model' :
            {'descr' : "Uniqsar/transformer_v1 molnet_muv benchmark model",
             'name' : 'molnet_muv',
             'tags' : ['uniqsar', 'qsar', 'gpu', 'heavy' 'chemformer']},
            
            'dataset' :
            {'name' : 'molnet_muv',
             'calc' : ['muv-466','muv-548','muv-600','muv-644','muv-652','muv-689','muv-692','muv-712','muv-713','muv-733','muv-737','muv-810','muv-832','muv-846' ,'muv-852','muv-858','muv-859'],
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
