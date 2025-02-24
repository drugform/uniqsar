import pymongo
import time

# mongodb:
# drugform -> [ users, generations ]
# generation_values -> [ task_1, task_2, ... ]
# generation_logs -> [ task_1, task_2, ... ]
#
# users = {user_id, limits, password, salt, email, name, init_date}
# generations = {user_id, generator_params, task, task_id, is_finished, init_date}

test_user = {'user_id' : 'test',
             'password' : 'test',
             'salt' : '',
             'limits' : {},
             'email' : '',
             'name' : 'Test user',
             'init_date' : 0}

class GenDB ():
    def __init__ (self, task_id, user_id, db_address, queue_limit=100):
        self.read_only = True
        self.db_address = db_address
        self.queue_limit = queue_limit
        self.task_id = task_id
        self.user_id = user_id
        self.client = pymongo.MongoClient(self.db_address)
        self.queues = {}
        
    def init_generation (self, generator_params, task, task_info):
        if '.' in self.task_id:
            raise Exception(f'Dot char `.` not allowed in task_id')
        drugform_db = self.client['drugform']
        users = drugform_db['users']
        generations = drugform_db['generations']
        user = users.find_one({'user_id' : self.user_id})
        if user is None:
            raise Exception(f'Unknown user: {self.user_id}')

        if generations.find_one({'task_id' : self.task_id}) is not None:
            raise Exception(f'Task {self.task_id} already exists')
        
        n_user_gens = generations.count_documents({'user_id' : self.user_id})
        if 'max_generations' in user['limits'].keys() and \
           n_user_gens >= user['limits']['max_generations']:
            raise Exception(f'User {self.user_id} reached `max_generations` limit')
            
        generations.insert_one({
            'task_id' : self.task_id,
            'user_id' : self.user_id,
            'generator_params' : generator_params,
            'task' : task,
            'task_info' : task_info,
            'start_time' : int(time.time()),
            'end_time' : None})
        
        self.read_only = False

    def __enter__ (self):
        return self

    def __exit__ (self, exception_type, exception_value, exception_traceback):
        if not self.read_only:
            self.flush_queues()
            self.client['drugform']['generations'].update_one(
                {'task_id' : self.task_id},
                {'$set' : {'is_finished' : True}})
        
        self.client.close()

    def flush_queues (self):
        for key, q in self.queues.items():
            db_name, col_name = key.split('.')
            self.client[db_name][col_name].insert_many(q)

        self.queues = {}
        
    def put (self, db_name, col_name, obj):
        key = f"{db_name}.{col_name}"
        if self.queues.get(key) is None:
            self.queues[key] = []

        self.queues[key].append(obj)
        q = self.queues[key]
        if len(q) > self.queue_limit:
            self.client[db_name][col_name].insert_many(q)
            self.queues[key] = []

    def get (self, db_name, col_name, query=None, sort_key=None, top_k=None):
        cursor = self.client[db_name][col_name].find(query)
        if top_k is not None:
            cursor = cursor.limit(top_k)
            
        if sort_key is not None:
            cursor = cursor.sort(sort_key, pymongo.DESCENDING)

        records = list(cursor)
        for rec in records:
            del(rec["_id"])

        return records

    
