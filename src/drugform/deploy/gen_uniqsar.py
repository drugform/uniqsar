import os
import yaml            

def compose_template (model_name, model_info, model_port):
    devs = model_info['devices']
    if type(devs) is int:
        device_list = [ str(devs) ]
    elif type(devs) is str:
        device_list = devs.split(',')
    else:
        raise Exception(f"Invalid device: {devs}")
    
    compose_template = f'''
  {model_name}:
    image: deploy-uniqsar
    command: python -u run.py
    network_mode: host
    volumes:
      - ../../uniqsar/models/:/app/models
      - ../../uniqsar/data/:/app/data
    environment:
      - DF_REGISTRY_ENDPOINT=tcp://${{CPU_HOST}}:${{REGISTRY_PORT}}
      - DF_SERVICE_ENDPOINT=tcp://${{GPU_HOST_1}}:{model_port}
      - DF_DB_ADDRESS=mongodb://${{CPU_HOST}}:${{DB_PORT}}
      - DF_MODEL_NAME={model_name}
      - MKL_NUM_THREADS=4
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: {device_list}
              capabilities: [gpu]'''
    return compose_template

def generate_compose (model_list, compose_file="docker-compose-uniqsar.yml"):
    start_port = 400101 # 400100 reserved for uniqsar orchestrator
    with open(compose_file, 'w') as fp:
        fp.write('version: "3"\n\nservices:')

        for i,(name,info) in enumerate(model_list.items()):
            block = compose_template(name, info, start_port)
            fp.write(block)
            start_port += 1
        
    return compose_file 

def read_model_list (fname='uniqsar_models.txt'):
    if not os.path.exists(fname):
        print(f'Models list file {fname} does not exist')
        return []
        
    with open(fname, "r") as fp:
        model_list = yaml.safe_load(fp)

    return model_list

if __name__ == "__main__":
    generate_compose(
        read_model_list())
    print('Compose config generated')
