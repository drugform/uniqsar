import requests
import os
import json
import time
import sys

backend_address = os.environ['DF_BACKEND_ADDRESS']

encode_url = os.path.join(backend_address, 'encode_mol')
calc_url = os.path.join(backend_address, 'calc')
gen_url = os.path.join(backend_address, 'generate')
gen_fetch_url = os.path.join(backend_address, 'generate_fetch')
list_services_url = os.path.join(backend_address, 'list_services')

def gen_fetch (task_id, top_k):
    r = requests.post(gen_fetch_url, json={"task_id" : task_id,
                                           "top_k" : int(top_k)})
    try:
        print(json.dumps(json.loads(r.text), indent=2))
    except:
        print(r.text)

if __name__ == "__main__":
    task_id = sys.argv[1]
    top_k = sys.argv[2]
    gen_fetch(task_id, top_k)
