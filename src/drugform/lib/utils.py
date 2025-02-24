import gevent.pool
import numpy as np
import decimal
import zlib
import base58

# сравнить корректность и эффективность методов
# Внимание - это не параллельность, а асинхронность!
def p_map_alt (fn, args):
    p = gevent.pool.Pool()
    ma = p.map_async(fn, args)
    ma.join()
    return ma.value

def p_map(f, iterable):
    group = gevent.pool.Group()
    ret = group.map(f, iterable)
    return ret

################################################################

def short_hash (src_string):
    hash_int = zlib.crc32(bytes(src_string.encode()))
    hash_str = base58.b58encode_int(hash_int).decode()
    return hash_str

################################################################

# obsolete
def normalize_value (value, minval, maxval):
    val = np.clip(value, minval, maxval)
    score = (val-minval)/(maxval-minval)
    return score

# obsolete
def apply_aim_modifier (prop_name, score, params):
    aim = params['aim']
    if aim == 'maximize':
        pass
    elif aim == 'minimize':
        score = 1-score
    elif aim == 'calc':
        score = None
    else:
        raise Exception(f'Unknown aim modifier: {params["aim"]}')

    return score
                
################################################################

def calc_score (value, prop_info, prop_params):
    minval, maxval = prop_info['range']
    norm_val = normalize_value(value, minval, maxval)

    aim = prop_params['aim']
    if aim == 'maximize':
        score = norm_val
    elif aim == 'minimize':
        score = 1-norm_val
    elif aim == 'calc':
        score = None
    elif aim == 'exact':
        tgt = prop_params['target']
        score = int(val==tgt)
    elif aim == 'value':
        tgt = prop_params['aim_value']
        norm_tgt = normalize_value(tgt, minval, maxval)
        dist = abs(norm_val - norm_tgt)
        norm_dist = dist / max(norm_tgt, 1-norm_tgt)
        score = 1-norm_dist
        
    elif aim == 'range':
        tgt_min, tgt_max = prop_params['range']
        norm_tgt_min = normalize_value(tgt_min, minval, maxval)
        norm_tgt_max = normalize_value(tgt_max, minval, maxval)
        if norm_val < norm_tgt_min:
            dist = norm_val - norm_tgt_min
        elif norm_val > norm_tgt_max:
            dist = norm_tgt_max-norm_val
        else:
            dist = 0
        norm_dist = dist / max(norm_tgt_min, 1-norm_tgt_max)
        score = 1-norm_dist

    if score is not None:
        score = round_fp(score, 3)
    return score
        

################################################################

def round_fp (num, prec):
    # обходим эпическое округление в питоне
    return float(round(decimal.Decimal(num), prec))

def round_fmt (num, prec):
    return round(decimal.Decimal(num), prec)

################################################################

def batch_split (data, batch_size):
    i = 0
    while i < len(data):
        yield data[i:min(i+batch_size, len(data))]
        i += batch_size

################################################################
