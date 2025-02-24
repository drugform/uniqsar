import zerorpc
import msgpack_numpy
msgpack_numpy.patch()

def RPCall (endpoint, remote_method, *args, **kwargs):
    c = zerorpc.Client()
    c.connect(endpoint)
    try:
        ret = c.__call__(remote_method, *args, **kwargs)
        c.close()
        return ret
    except Exception as e:
        c.close()
        raise Exception(e)

class RPConn ():
    def __init__ (self, endpoint):
        self.c = zerorpc.Client()
        self.c.connect(endpoint)

    def __call__ (self, remote_method, *args, **kwargs):
        return self.c.__call__(remote_method, *args, **kwargs)

    def close (self):
        return self.c.close()
    
