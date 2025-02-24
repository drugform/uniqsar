import os
from registry import RegistryApp

if __name__ == "__main__":
    registry_endpoint = os.environ['DF_REGISTRY_ENDPOINT']
    app = RegistryApp(registry_endpoint)
    app.run()
