import decimal

def round_fp (num, prec):
    return float(round(decimal.Decimal(num), prec))

def round_fmt (num, prec):
    return round(decimal.Decimal(num), prec)

def batch_split (data, batch_size):
    i = 0
    while i < len(data):
        yield data[i:min(i+batch_size, len(data))]
        i += batch_size
