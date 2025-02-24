import yaml
import zerorpc

def tee (obj):
    print(obj)
    return obj

def load_config (path):
    with open(path, 'r') as fh:
        data = fh.read()
    
    cfg = yaml.load(data, Loader=yaml.CLoader)
    return cfg

class RegistryApp (object):
    def __init__ (self, endpoint):
        self.services = {}
        self.endpoint = endpoint
        self.read_props()
        self.run()

    def read_props (self):
        props_file = 'props.cfg'
        props = load_config(props_file)
        for name,params in props.items():
            if params.get('unit') is None:
                props[name]['unit'] = ''

        self.props = props
        
    def register_service (self, key, service_info):
        self.update_services()
        if self.services.get(key) is not None:
            return tee({'status'  : 'error',
                        'message' : f'service {key} already exists'})

        all_endpoints = [info['endpoint'] for info
                         in self.services.values()]
        if service_info['endpoint'] in all_endpoints:
            return tee({'status' : 'error',
                        'message' : f'endpoint {self.endpoint} already in use'})
        
        for prop in service_info['props']:
            if prop not in self.props.keys():
                return tee({'status' : 'error',
                            'message' : f'unknown prop: {prop}'})
        
        self.services[key] = service_info
        return {'status' : 'success'}

    def get_props_info (self):
        return self.props
    
    def delete_service (self, key):
        del self.services[key]
    
    def list_services (self):
        return self.services

    def update_services (self):
        for key,info in list(self.services.items()):
            endpoint = info['endpoint']
            c = zerorpc.Client()
            c.connect(endpoint)
            try:
                assert c.ping() == 'pong'
            except:
                print(f'deleting service {key}:{endpoint}')
                self.delete_service(key)
            finally:
                c.close()
    
    def run (self):
        s = zerorpc.Server(self)
        s.bind(self.endpoint)
        print(f'Registry started at {self.endpoint}')
        s.run()
        
