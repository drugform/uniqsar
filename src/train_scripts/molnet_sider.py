from model import Model

sider_props = ['hepatobiliary_disorders', 'metabolism_and_nutrition_disorders', 'product_issues', 'eye_disorders', 'investigations', 'musculoskeletal_and_connective_tissue_disorders', 'gastrointestinal_disorders', 'social_circumstances', 'immune_system_disorders', 'reproductive_system_and_breast_disorders', 'neoplasms_benign,_malignant_and_unspecified_(incl_cysts_and_polyps)', 'general_disorders_and_administration_site_conditions', 'endocrine_disorders', 'surgical_and_medical_procedures', 'vascular_disorders', 'blood_and_lymphatic_system_disorders', 'skin_and_subcutaneous_tissue_disorders', 'congenital,_familial_and_genetic_disorders', 'infections_and_infestations', 'respiratory,_thoracic_and_mediastinal_disorders', 'psychiatric_disorders', 'renal_and_urinary_disorders', 'pregnancy,_puerperium_and_perinatal_conditions', 'ear_and_labyrinth_disorders', 'cardiac_disorders', 'nervous_system_disorders', 'injury,_poisoning_and_procedural_complications']

if __name__ == "__main__":
    device = 'cuda:0'
    task = {'model' :
            {'descr' : "Uniqsar/transformer_v1 molnet_sider benchmark model",
             'name' : 'molnet_sider',
             'tags' : ['uniqsar', 'qsar', 'gpu', 'heavy' 'chemformer']},
            
            'dataset' :
            {'name' : 'molnet_sider',
             'calc' : sider_props,
             'deriv' : {},
             'param' : []},
            
            'encoder' :
            {'name' : 'chemformer',
             'variant' : 'light',
             'augment' : 10,
             'minpad' : 0},
            
            'net' :
            {'name' : 'transformer_v1',
             'n_layers' : 6},
            
            'train' :
            {'batch_size' : 64,
             'n_folds' : 5,
             'train_prop' : 0.9,
             'learning_rate' : 3e-4,
             'n_epochs' : 150,
             'n_best_nets' : 5,
             'tune' : True}}
    
    m = Model(task, device)
