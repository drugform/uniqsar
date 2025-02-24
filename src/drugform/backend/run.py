import os
from server import ServerApp

if __name__ == "__main__":
    registry_endpoint = os.environ['DF_REGISTRY_ENDPOINT']
    backend_address = os.environ['DF_BACKEND_ADDRESS']
    db_address = os.environ['DF_DB_ADDRESS']
    debug = os.environ.get('DF_BACKEND_DEBUG', False)
    app = ServerApp(backend_address, registry_endpoint, db_address, debug)
    app.run()
