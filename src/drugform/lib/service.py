from threading import Thread
import zerorpc

import pymongo

from drugform.lib.mol import Mol
from drugform.lib.rpc import RPCall
from drugform.lib.utils import short_hash

class ServiceApp ():
    def __init__ (self, model, tags,
                  endpoint, registry_endpoint):
        self.model = model
        self.tags = tags
        self.endpoint = endpoint
        self.registry_endpoint = registry_endpoint
        self.registry_info = {'props' : model.props,
                              'model' : model.name,
                              'tags' : tags,
                              'endpoint' : endpoint}

        self.model.props_info = RPCall(self.registry_endpoint, 'get_props_info')
        
    def run (self):
        self.server = zerorpc.Server(self)
        self.server.bind(self.endpoint)
        self.register()
        print(f'Service "{self.model.name}" started at {self.endpoint}')
        self.server.run()
        
    def register (self):
        reg = zerorpc.Client()
        reg.connect(self.registry_endpoint)
        key = self.model.name
        ret = reg.register_service(key, self.registry_info)
        reg.close()
        if ret['status'] == 'error':
            raise Exception(ret['message'])

    def ping (self):
        return 'pong'

class CalcServiceApp (ServiceApp):
    def calc (self, pakmol, task_dict, task_id=None):
        mols = [Mol.unpack(p) for p in pakmol]
        return self.model(mols, task_dict, task_id)

class EncoderServiceApp (ServiceApp):
    def encode (self, samples, task_dict):
        return self.model(samples, task_dict)
    
class GeneratorServiceApp (ServiceApp):
    # переделать Thread -> Process
    def generate (self, generator_params, task, user_id, task_info):
        if not hasattr(self, 'workers'):
            self.workers = []
            self.max_workers = 4 # fixme

        for wid, worker in enumerate(self.workers):
            if not worker.is_alive():
                worker.join()
                del self.workers[wid]

        if len(self.workers) >= self.max_workers:
            raise Exception(f'Too many tasks currently running: {len(self.workers)}')

        task_id = self.make_task_id(task_info['name'])
        def fn ():
            self.model.run(generator_params, task, user_id, task_info, task_id)
        
        new_worker = Thread(target=fn, args=(), daemon=True)
        new_worker.start()
        self.workers.append(new_worker)
        return task_id

    def make_task_id (self, name): # в utils?
        task_key = short_hash(name)
        cli = pymongo.MongoClient(self.model.db_address)
        match_count = 0
        for gen in cli['drugform']['generations'].find():
            if task_key in gen['task_id']:
                match_count += 1

        task_id = f"{task_key}_{match_count}"
        return task_id
                    
    def stop (self, task_id):
        self.model.stop(task_id)
        return {'status' : 'success'}
