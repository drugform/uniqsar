import requests
import socket
import os
import sys
import json

def get_backend_address (hostname, port):
    ip = socket.gethostbyname(hostname)
    address = f"http://{ip}:{port}"
    return address

def make_url (hostname, route, port=40001):
    backend_address = get_backend_address(hostname, port)
    return backend_address + route

def get_payload (resp):
    status = resp.status_code
    if status == 401:
        raise Exception(f'Authorization required')
    
    elif status != 200:
        raise Exception(f"Server returned {status}")

    ret = resp.json()
    if not ret['success']:
        raise Exception(ret['message'])
    return ret['data']

def read_json (fname):
    with open(fname, 'r') as fp:
        return json.load(fp)
