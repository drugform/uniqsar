from model import Model

prime_numbers = [1,2,3,5,7,11,13,17,19,23,29,31,37]
size_filters = prime_numbers[:9]
n_filters = [64]*len(size_filters)
filters = list(zip(n_filters, size_filters))
minpad=max(size_filters)

tox_props = {"o_mus_ipr_ld": (2.92, 0.0),
             "o_rat_orl_tdlo": (0.46, -1.0),
             "o_mus_ipr_ldlo": (0.83, -0.0),
             "o_mus_orl_tdlo": (0.38, -1.2),
             "o_rat_ipr_tdlo": (0.48, -1.4),
             "o_mus_ivn_ld": (1.99, -0.5),
             "o_rat_ipr_ld": (1.05, -0.0),
             "o_mus_orl_ld": (2.36, 0.4),
             "o_mus_unr_ld": (0.65, 0.1),
             "o_rat_unr_ld": (0.44, 0.1),
             "o_mus_scu_ldlo": (0.45, -0.1),
             "o_rat_scu_ld": (0.63, 0.4),
             "o_mus_scu_ld": (1.25, 0.3),
             "o_rat_ipr_ldlo": (0.48, -0.1),
             "o_mus_ipr_tdlo": (0.48, -1.4),
             "o_rbt_skn_ld": (0.64, 0.8),
             "o_rat_orl_ld": (1.55, 0.5),
             "o_rat_ivn_tdlo": (0.35, -1.9),
             "o_rat_orl_ldlo": (0.47, 0.2),
             "o_rbt_orl_ld": (0.45, 0.5),
             "o_rbt_ivn_ld": (0.38, -0.8),
             "o_rat_ivn_ld": (0.71, -0.6),
             "o_mus_orl_ldlo": (0.6, 0.3),
             "o_rat_skn_ld": (0.47, 0.8),
             "o_mam_unr_ld": (0.51, 0.2),
             "o_gpg_orl_ld": (0.42, 0.3),
             "o_wmn_orl_tdlo": (0.31, -1.0),
             "o_man_orl_tdlo": (0.32, -1.1),
             "o_rat_scu_tdlo": (0.33, -1.8) }


if __name__ == "__main__":
    device = 'cuda:0'
    task = {'model' :
            {'descr' : "Uniqsar/textcnn toxicity model",
             'name' : 'toxicity_v2',
             'tags' : ['uniqsar', 'qsar', 'gpu', 'chemformer']},
            
            'dataset' :
            {'name' : 'toxicity',
             'calc' : list(tox_props.keys()),
             'deriv' : {'toxicity' : tox_props},
             'param' : []},
            
            'encoder' :
            {'name' : 'chemformer',
             'variant' : 'light',
             'augment' : 10,
             'minpad' : minpad},
            
            'net' :
            {'name' : 'textcnn',
             'filters' : filters,
             'mlp_dim' : 128,
             'dropout' : 0.2,
             'activation' : 'relu'},
            
            'train' :
            {'batch_size' : 64,
             'n_folds' : 6,
             'train_prop' : 0.9,
             'learning_rate' : 1e-3,
             'n_epochs' : 150,
             'n_best_nets' : 3,
             'tune' : True}}
    
    m = Model(task, device)
