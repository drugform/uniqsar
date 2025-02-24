from model import Model

pcba_props = ["pcba-1030","pcba-1379","pcba-1452","pcba-1454","pcba-1457","pcba-1458","pcba-1460","pcba-1461","pcba-1468","pcba-1469","pcba-1471","pcba-1479","pcba-1631","pcba-1634","pcba-1688","pcba-1721","pcba-2100","pcba-2101","pcba-2147","pcba-2242","pcba-2326","pcba-2451","pcba-2517","pcba-2528","pcba-2546","pcba-2549","pcba-2551","pcba-2662","pcba-2675","pcba-2676","pcba-411","pcba-463254","pcba-485281","pcba-485290","pcba-485294","pcba-485297","pcba-485313","pcba-485314","pcba-485341","pcba-485349","pcba-485353","pcba-485360","pcba-485364","pcba-485367","pcba-492947","pcba-493208","pcba-504327","pcba-504332","pcba-504333","pcba-504339","pcba-504444","pcba-504466","pcba-504467","pcba-504706","pcba-504842","pcba-504845","pcba-504847","pcba-504891","pcba-540276","pcba-540317","pcba-588342","pcba-588453","pcba-588456","pcba-588579","pcba-588590","pcba-588591","pcba-588795","pcba-588855","pcba-602179","pcba-602233","pcba-602310","pcba-602313","pcba-602332","pcba-624170","pcba-624171","pcba-624173","pcba-624202","pcba-624246","pcba-624287","pcba-624288","pcba-624291","pcba-624296","pcba-624297","pcba-624417","pcba-651635","pcba-651644","pcba-651768","pcba-651965","pcba-652025","pcba-652104","pcba-652105","pcba-652106","pcba-686970","pcba-686978","pcba-686979","pcba-720504","pcba-720532","pcba-720542","pcba-720551","pcba-720553","pcba-720579","pcba-720580","pcba-720707","pcba-720708","pcba-720709","pcba-720711","pcba-743255","pcba-743266","pcba-875","pcba-881","pcba-883","pcba-884","pcba-885","pcba-887","pcba-891","pcba-899","pcba-902","pcba-903","pcba-904","pcba-912","pcba-914","pcba-915","pcba-924","pcba-925","pcba-926","pcba-927","pcba-938","pcba-995"]

if __name__ == "__main__":
    device = 'cuda:0'
    task = {'model' :
            {'descr' : "Uniqsar/transformer_v1 molnet PCBA benchmark model",
             'name' : 'molnet_pcba',
             'tags' : ['uniqsar', 'qsar', 'gpu', 'chemformer',' heavy']},
            
            'dataset' :
            {'name' : 'molnet_pcba',
             'calc' : pcba_props,
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
             'learning_rate' : 1e-3,
             'n_epochs' : 40,
             'n_best_nets' : 5,
             'tune' : True}}
    
    m = Model(task, device)
