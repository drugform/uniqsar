from .rpc import RPCall
from .utils import p_map 

def parse_task (task):
    task_dict = {}
    aims = []
    for prop_req in task:
        model_name = prop_req['model']
        prop_name = prop_req['prop']
        if model_name not in task_dict:
            task_dict[model_name] = {}

        if prop_name in task_dict[model_name]:
            raise Exception(f"Duplicate property {prop_name} for model {model_name}")

        task_dict[model_name][prop_name] = prop_req['params']
        aims.append(prop_req['params']['aim'])

    if 'calc' in aims:
        if aims.count('calc') != len(aims):
            raise Exception('Mixed calc and non-calc aims not allowed')

    return task_dict

def calc_task (pakmol, task_dict, services, task_id):
    model_names = list(task_dict.keys())

    def calc_model (model_name):
        prop_dict = task_dict[model_name]
        endpoint = services[model_name]['endpoint']
        model_ret = RPCall(endpoint, 'calc',
                           pakmol, prop_dict, task_id)
        return model_ret
                
    model_rets = p_map(calc_model, model_names)
    result = format_calc(pakmol, task_dict, model_rets)
    return result


def format_calc (pakmol, task_dict, model_rets):
    result = [dict() for _ in pakmol]
    scores = [{'sum' : 1, 'w' : 0} for _ in pakmol]
    calc_only = None
    model_names = list(task_dict.keys())
    
    for model_id, model_name in enumerate(model_names):
        for mol_id, mol_ret in enumerate(model_rets[model_id]):
            for prop_name, prop_value in mol_ret.items():
                if result[mol_id].get(prop_name) is None:
                    result[mol_id][prop_name] = {}

                result[mol_id][prop_name][model_name] = prop_value
                prop_task = task_dict[model_name][prop_name]
                if calc_only is None:
                    calc_only = (prop_task['aim'] == 'calc')

                if not calc_only:
                    prop_weight = prop_task['weight']
                    prop_score = prop_value['score']
                    #scores[mol_id]['sum'] += prop_weight * prop_score
                    #scores[mol_id]['w'] += prop_weight
                    scores[mol_id]['sum'] *= (prop_score ** prop_weight)
                    scores[mol_id]['w'] += prop_weight
                            
    if not calc_only:
        for mol_id in range(len(pakmol)):
            #total_score = scores[mol_id]['sum'] / scores[mol_id]['w']
            total_score = scores[mol_id]['sum'] ** (1/scores[mol_id]['w']) # geometric mean
            result[mol_id]['total'] = {'score' : round(total_score, 3)}

    return result

def check_models (registry_endpoint, task_dict):
    services = RPCall(registry_endpoint, 'list_services')
    for model_name in task_dict:
        if model_name not in services:
            raise Exception(f'Model {model_name} is not running')
        for prop_name in task_dict[model_name]:
            if prop_name not in services[model_name]['props']:
                raise Exception(f'Model {model_name} not providing property {prop_name}')

    return services

def check_service (registry_endpoint, name):
    services = RPCall(registry_endpoint, 'list_services')
    if name not in services:
        raise Exception(f'Service {name} not running')
    return services[name]
    
