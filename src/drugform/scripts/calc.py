import os
import sys
import requests

import utils

def encode_csv (hostname, input_file):
    url = utils.make_url(hostname, '/api/encodeCSV')
    with open(input_file, 'rb') as fp:
        resp = requests.post(url,
                             auth=('test','test'),
                             files={'file': fp})

    mols = utils.get_payload(resp)
    return mols

def calc (hostname, mols, task_params):
    url = utils.make_url(hostname, '/api/calc')
    resp = requests.post(url,
                         auth=('test','test'),
                         json={"mols" : mols,
                               "taskParams" : task_params})
    results = utils.get_payload(resp)
    return results
    
    
def make_csv (hostname, mols, results):
    url = utils.make_url(hostname, '/api/makeCSV')
    resp = requests.post(url,
                         auth=('test','test'),
                         json={"mols" : mols,
                               "results" : results})
    csv_text = utils.get_payload(resp)
    return csv_text

def run (hostname, input_file, task_file):
    mols = encode_csv(hostname, input_file)
    task_params = utils.read_json(task_file)
    results = calc(hostname, mols, task_params)
    csv_text = make_csv(hostname, mols, results)

    input_name, ext = os.path.splitext(input_file)
    output_file = f"{input_name}_out{ext}"
    with open(output_file, 'w') as fp:
        fp.write(csv_text)

    print(f'<<< {output_file} written')
    
if __name__ == "__main__":
    hostname = sys.argv[1]
    input_file = sys.argv[2]
    task_file = sys.argv[3]
    run(hostname, input_file, task_file) 
    
