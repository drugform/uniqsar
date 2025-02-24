import os
import csv
import json
import numpy as np
from copy import deepcopy
import time 
import traceback
from base58 import b58encode, b58decode

from rdkit import Chem # move to mol with drawing code
from rdkit.Chem import Draw
from io import BytesIO
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from drugform.lib.mol import Mol
from drugform.lib.rpc import RPCall, RPConn
from drugform.lib.utils  import p_map
from drugform.lib.calc import calc_task, check_models, check_service, parse_task
from drugform.lib.db import GenDB

import tempfile
from flask import Flask, request, make_response, redirect, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS
from waitress import serve
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

def print_traceback (e):
    return '\n'.join(traceback.TracebackException.from_exception(e).format())

def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=90)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

def moving_average (values, w):
    y = np.convolve(np.array(values), np.ones(w), 'valid')/w
    #y = savgol_filter(values, w, 1, mode='nearest')
    return y

def draw_metrics_img (metrics):
    "Temporary code"
    use_keys = ["Score (cur.avg)",
                "Score (iter top)",
                "Loss (prior)",
                "Loss (total)"]
    x_key = 'step'
    values = [[],[],[],[]]
    x = []
    for record in metrics:
        x.append(record[x_key])
        for i in range(4):
            values[i].append(record[use_keys[i]])

    fig = plt.figure(figsize=(16,10))
    gs = fig.add_gridspec(4, hspace=0)
    axs = gs.subplots(sharex=True)
    
    for i,ax in enumerate(axs.flat):
        ax.grid(True)
        ax.plot(x, values[i], alpha=0.33)
        ax.plot(x[10-1:], moving_average(values[i], 10), alpha=0.67)
        ax.plot(x[100-1:], moving_average(values[i], 100), alpha=1)
        ax.set_title(use_keys[i], x=0.05, y=0.8)
        ax.set(xlabel='Step')
        #ax.label_outer()

    fig.tight_layout()
    fig.canvas.draw()
    img = Image.frombytes('RGB',
                          fig.canvas.get_width_height(),
                          fig.canvas.tostring_rgb())
    return img

UPLOAD_RECORD_LIMIT = 100000
UPLOAD_MAX_N_ATOMS = 100

def read_smiles_from_csv_future (filepath):
    df = pd.read_csv(filepath)
    if ('smiles' in df.columns) and ('SMILES' in df.columns):
        raise Exception('File must contain a single SMILES column')
    if 'smiles' in df.columns:
        smiles = df.smiles.tolist()
    elif 'SMILES' in df.columns:
        smiles = df.SMILES.tolist()
    else:
        raise Exception('File must contain a SMILES column')

    pakmol, fails = [],[]
    for i,sm in enumerate(smiles):
        if len(pakmol) >= UPLOAD_RECORD_LIMIT:
            fails.append([i, 'upload_limit', (len(pakmol), UPLOAD_RECORD_LIMIT)])
            continue
            
        try:
            mol = Mol(sm)
            if mol.rdmol.GetNumHeavyAtoms() > UPLOAD_MAX_N_ATOMS:
                fails.append([i, 'too_many_atoms', (mol.rdmol.GetNumHeavyAtoms(), UPLOAD_MAX_N_ATOMS)])
                continue
                
            pakmol.append(mol.pack())
        except Exception as e:
            fails.append([i, 'bad_smiles', str(e)])

    return pakmol, fails

def read_smiles_from_csv (filepath):
    df = pd.read_csv(filepath)
    if ('smiles' in df.columns) and ('SMILES' in df.columns):
        raise Exception('File must contain a single SMILES column')
    if 'smiles' in df.columns:
        smiles = df.smiles.tolist()
    elif 'SMILES' in df.columns:
        smiles = df.SMILES.tolist()
    else:
        raise Exception('File must contain a SMILES column')

    if len(smiles) > UPLOAD_RECORD_LIMIT:
        raise Exception(f'File size exceeds limit ({UPLOAD_RECORD_LIMIT})')
    
    pakmol = []
    for i,sm in enumerate(smiles):
        try:
            mol = Mol(sm)
            if mol.rdmol.GetNumHeavyAtoms() > UPLOAD_MAX_N_ATOMS:
                raise Exception(f'Molecule {i} exceeds heavy atom number limit ({UPLOAD_MAX_N_ATOMS})')
            pakmol.append(mol.pack())
        except:
            raise Exception(f'Failed to encode SMILES: {sm}')

    return pakmol

def make_csv_ (pakmol, results):
    def make_header_info (res):
        header = []
        for prop,val in res.items():
            if prop == 'total':
                continue
            
            models = list(val.keys())
            if len(models) == 1:
                header.append((prop,
                               prop,
                               models[0]))
            else:
                for model in models:
                    header.append((f"{prop}/{model}",
                                   prop,
                                   model))
        return header

    def make_header_row (header_info):
        row = ['id', 'SMILES', 'score']
        for prop_name,_,_ in header_info:
            row.append(f"{prop_name}/value")
            row.append(f"{prop_name}/score")
        return row
    
    header_info = make_header_info(results[0])
    csv_rows = [','.join(make_header_row(header_info))]
    for i,(mol,res) in enumerate(zip(pakmol, results)):
        row = [str(i),
               mol['smiles'],
               str(res['total']['score'])]
        for prop_name, prop, model in header_info:
            value = res[prop][model]['value']
            score = res[prop][model]['score']
            
            row.append(str(value))
            row.append(str(score))
        csv_rows.append(','.join(row))
        
    csv_text = '\n'.join(csv_rows)
    return csv_text

def send_error (e, error, message):
    try: details = print_traceback(e)
    except: details = str(e)
    return json.dumps({'success' : False,
                       'error' : error,
                       'message' : f"{message}: {details}"})

def send_result (ret, aux=None):
    return json.dumps({'success' : True,
                       'data' : ret,
                       'aux' : aux})

class ServerApp ():
    def __init__ (self, backend_address, registry_endpoint, db_address, debug):
        self.registry_endpoint = registry_endpoint
        self.backend_address = backend_address
        self.db_address = db_address
        self.backend_port = int(backend_address.split(':')[-1])
        self.debug = debug

    def run (self):
        app = Flask(__name__,
                    template_folder='www',
                    static_folder='www')
        app.debug = self.debug
        app.config['UPLOAD_FOLDER'] = 'upload'
        app.config['MAX_CONTENT_LENGTH'] = 100*1024*1024 # 100MB
        #CORS(app)

        auth = HTTPBasicAuth()
        users = {"test" : generate_password_hash("test")}

        @auth.verify_password
        def verify_password (username, password):
            if username in users and \
               check_password_hash(users.get(username), password):
                return username

        @app.route('/image/mol/<string:imgenc>', methods=['GET'])
        def draw_mol (imgenc):
            mol = Chem.Mol(b58decode(imgenc))
            n_atoms = mol.GetNumHeavyAtoms()
            height = 150 + 15*n_atoms
            width = int(height*4/3)
            img = Draw.MolToImage(mol,
                                  kekulize=True,
                                  size=(width, height))

            return serve_pil_image(img)

        @app.route('/api/generate/<string:taskId>/metrics', methods=['GET'])
        @auth.login_required
        def draw_metrics (taskId):
            task_id = taskId # fixme
            user_id = auth.current_user()
            with GenDB(task_id, user_id, self.db_address) as db:
                # check task access
                generations = db.get('drugform', 'generations',
                                     query={'task_id' : task_id,
                                            'user_id' : user_id})
                if len(generations) == 0:
                    return send_error(f"task_id={task_id}:user_id={user_id}",
                                      "Task not found",
                                      'No task with given taskId')
                
                gen_info = generations[0]
                
            with GenDB(task_id, user_id, self.db_address) as db:
                metrics = db.get('generation_logs', task_id)
                img = draw_metrics_img(metrics)
                return serve_pil_image(img)
        
        @app.route('/api/encodeMol', methods=['POST'])
        def encode_mol ():
            try:
                smiles = request.json['smiles']
                return send_result(Mol(smiles).pack())
            except Exception as e:
                return send_error(e, "Invalid SMILES", "Unable to parse SMILES")    

        @app.route('/api/encodeCSV', methods=['POST'])
        @auth.login_required
        def encode_csv ():
            try:
                file = request.files['file']
                filename = secure_filename(file.filename)
                tmpdir = tempfile.mkdtemp()
                filepath = os.path.join(tmpdir, 'file.csv')
                file.save(filepath)
                pakmol = read_smiles_from_csv(filepath)
                
                try: os.remove(filepath)
                except: pass

                return send_result(pakmol)#, aux={'fails' : fails})
            except Exception as e:
                return send_error(e, "Invalid CSV file", "Failed to read molecules from CSV file")

        @app.route('/api/makeCSV', methods=['POST'])
        @auth.login_required
        def make_csv ():
            pakmol = request.json['mols']
            results = request.json['results']
            csv_text = make_csv_(pakmol, results)
            return send_result(csv_text)
            
        ################################################################

        @app.route('/api/calc', methods=['POST'])
        @auth.login_required
        def calc ():
            task = request.json['taskParams'] # task - список prop_dict
            pakmol = request.json['mols']
            #task = json.loads(task)
            try:
                task_dict = parse_task(task) # task_dict - {model:prop_dict}
            except Exception as e:
                return send_error(e, "Invalid task", "Failed to parse task parameters")
            try:
                services = check_models(self.registry_endpoint, task_dict)
            except Exception as e:
                return  send_error(e, "Requested model not available",
                           "Cannot reach one of requested models")

            try:
                result = calc_task(pakmol, task_dict, services, task_id=None)
                return send_result(result)
            except Exception as e:
                return send_error(e, "Unknown error", "Failed to calculate task")
                

        ################################################################
            
        @app.route('/api/generate', methods=['POST'])
        @auth.login_required
        def generate ():
            user_id = auth.current_user()
            task = request.json['taskParams']
            task_info = request.json['taskInfo']
            generator_params = request.json['generatorParams']

            #task = json.loads(task)
            #task_info = json.loads(task_info)
            #generator_params = json.loads(generator_params)
            
            try:
                task_dict = parse_task(task)
            except Exception as e:
                return send_error(e, "Invalid task",
                                  "Cannot parse task definition")

            try:
                generator_name = generator_params['name'] # name -> mode here and in transforcer
            except Exception as e:
                return send_error(e, "Invalid generator params",
                                  "Generator model not given")

            try:
                services = check_models(self.registry_endpoint, task_dict)
                generator_endpoint = check_service(self.registry_endpoint, 
                                               generator_name)['endpoint']
            except Exception as e:
                return send_error(e, "Requested model not available",
                                  "Cannot reach one of requested models")
                

            try:
                task_id = RPCall(generator_endpoint, 'generate',
                                 generator_params, task, user_id, task_info)
                return send_result({'taskId' : task_id})
            except Exception as e:
                return send_error(e, "Failed to run generator",
                                  "Error occured while running generation task")
                        
        #@app.route('/list_services', methods=['GET'])
        #def list_services ():
        #    services = RPCall(self.registry_endpoint, 'list_services')
        #    return services

        @app.route('/api/generate/<string:taskId>/info', methods=['GET', 'POST'])
        @auth.login_required
        def generate_task_info (taskId):
            task_id = taskId # check if can put task_id in func params
            user_id = auth.current_user()
            with GenDB(task_id, user_id, self.db_address) as db:
                generations = db.get('drugform', 'generations',
                                     query={'task_id' : task_id,
                                            'user_id' : user_id})
                if len(generations) == 0:
                    return send_error(f"task_id={task_id}:user_id={user_id}",
                                      "Task not found",
                                      'No task with given taskId')

                gen = generations[0]
                return send_result(
                    {"taskId": gen['task_id'],
                     "userId": gen['user_id'],
                     "generatorParams" : gen['generator_params'],
                     "taskParams" : gen['task'],
                     "taskInfo" : gen['task_info'],
                     "startTime" : gen['start_time'],
                     "endTime" : gen['end_time']})                

        @app.route('/api/generate/info', methods=['GET', 'POST'])
        @auth.login_required
        def generate_info ():
            user_id = auth.current_user()
            with GenDB(None, user_id, self.db_address) as db:
                generations = db.get('drugform', 'generations',
                                     query={'user_id' : user_id})
                task_ids = [gen['task_id'] for gen in generations]
                return send_result(task_ids)

        @app.route('/api/generate/<string:taskId>/results/<int:topK>', methods=['GET', 'POST'])
        @auth.login_required
        def generate_results (taskId, topK):
            task_id = taskId
            user_id = auth.current_user()
            with GenDB(task_id, user_id, self.db_address) as db:
                records = db.get('generation_values', task_id,
                                 sort_key='score', top_k=topK)
                for i,rec in enumerate(records):
                    sm = rec['smiles']
                    records[i]['mol'] = Mol(sm).pack()
            return send_result(records)

        @app.route('/api/generate/<string:taskId>/stop', methods=['POST'])
        @auth.login_required
        def generate_stop (taskId):
            user_id = auth.current_user()
            task_id = taskId
            with GenDB(task_id, user_id, self.db_address) as db:
                generations = db.get('drugform', 'generations',
                                     query={'task_id' : task_id,
                                            'user_id' : user_id})
                
                if len(generations) == 0:
                    return send_error(f"task_id={task_id}:user_id={user_id}",
                                      "Task not found",
                                      'No task with given taskId')

            gen = generations[0]
            if gen['end_time'] is not None:
                return send_error(f"finish time: {gen['end_time']}",
                                  "Task already stopped",
                                  "This task is already stopped")
            
            generator_name = gen['generator_params']['name']
            try:
                generator_endpoint = check_service(self.registry_endpoint, 
                                                  generator_name)['endpoint']
                RPCall(generator_endpoint, 'stop', task_id)
                return send_result()
            except Exception as e:
                return send_error(e, "No access to generator",
                                  "Required generation model is down. Task probably not running anyway")


        ################################################################
        serve(app, port=self.backend_port)
            
