import re
import sys
import shutil
import joblib
import pandas as pd
import numpy as np
import json
import requests
from collections import Counter
from tqdm import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings('ignore')
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

sys.path.append('..')
from utils import train_test_split, cold_drug_target_split

_cfg_ = {'pH_range' : [5.5, 8.5],
         'T_range' : [33, 40],
         'Kd/Ki_range' : [1e-2, 1e7],
         'Kd/Ki_percentile' : 20,
         'IC50_range' : [1e-2, 1e7],
         'IC50_percentile' : 20,
         'molwt_range' : [1e3, 5e5],
         'protein_len_range' : [50, 2560],
         'apprx_scale' : 3.16}


def read_bindingdb (fname):
    print('Reading BindingDB')
    columns = {'Ligand SMILES' : 'smiles',
               'Ligand InChI' : 'inchi',
               'Ki (nM)' : 'Ki',
               'IC50 (nM)' : 'IC50',
               'Kd (nM)' : 'Kd',
               'pH' : 'pH',
               'Temp (C)' : 'T',
               'Link to Target in BindingDB' : 'bdb_link',
               'BindingDB Target Chain Sequence' : 'sequence',
               'Number of Protein Chains in Target (>1 implies a multichain complex)' : 'n_chains',
               'Target Source Organism According to Curator or DataSource' : 'organizm',
               "UniProt (SwissProt) Primary ID of Target Chain" : 'uniprot_id',
               'UniProt (SwissProt) Entry Name of Target Chain' : 'uniprot_name'}

    df = pd.read_csv(fname,
                     sep='\t',
                     usecols=list(columns.keys()))
    df = df.rename(columns=columns)
    df['weight'] = np.ones(len(df))
    df['bdb_link'] = df['bdb_link'].apply(lambda url: 'Increment=1'.join(url.split('Increment=50')))
    return df

def fill_missing_uniprot_ids (bdb):
    urls = collect_missing_uniprot_ids(bdb)
    def url2uid (url):
        uid = urls.get(url)
        if uid is None:
            print(f'Uniprot ID for {url} still missing')
            return np.nan
        else:
            return uid
    1
    miss_part = bdb[bdb['uniprot_id'].isna()]
    bdb.loc[bdb['uniprot_id'].isna(), 'uniprot_id'] = miss_part['bdb_link'].apply(url2uid)
        

def collect_missing_uniprot_ids (bdb):
    fname = 'missing_uniprot_ids.pkl'
    urls = joblib.load(fname)
    miss_ids = list(bdb[bdb['uniprot_id'].isna()].index)
    1
    def parse_uniprot_id (seq):
        com_pos = seq.find(',')
        br_pos = seq.find('[')
        if com_pos == -1:
            return seq
        elif br_pos == -1:
            return seq.split(',')[0]
        elif com_pos < br_pos:
            return seq.split(',')[0]
        else:
            return seq.split(']')[0]+']'
    1
    for i in tqdm(miss_ids):
        row = bdb.iloc[i]
        bdb_link = row['bdb_link']
        if bdb_link in urls:
            continue
        1
        print(i, bdb_link)
        ret = requests.get(bdb_link)
        if not ret.ok:
            print(bdb_link, ret.status_code)
            continue
        1
        page = ret.text
        pos_sw = page.find('UniProtKB/SwissProt')
        pos_tr = page.find('UniProtKB/TrEMBL')
        if pos_sw != -1:
            pos = pos_sw
        elif pos_tr != -1:
            pos = pos_tr
        else:
            print('Key not found')
        try:
            req_start = page.rfind('forward_otherdbs.jsp?dbName=UniProt',0,pos)
            if req_start == -1:
                raise Exception('no uniprot record available')
            ids_start = page.find('ids=', req_start, pos)
            seq = page[ids_start+4:pos]
            uniprot_seq = seq.split('&')[0].split('"')[0].replace(' ','')
            uniprot_id = parse_uniprot_id(uniprot_seq)
            #uniprot_id = uniprot_seq.split('"')[0].split(',')[0]
            print(f'got {uniprot_id}')
            urls[bdb_link] = uniprot_id
            joblib.dump(urls, fname)
        except Exception as e:
            print(e)
    1
    return urls

def read_uniprot (fname):
    print('Reading UniProt')
    columns = {'accession_number' : 'uniprot_id',
               'entry_name' : 'uniprot_name',
               'recommended_name' : 'recommended_name',
               'organism_name' : 'organism_name',
               'organism_id' : 'organism_id',
               'sequence' : 'sequence'}
    
    df = pd.read_csv(fname,
                     usecols=list(columns.keys()))
    df = df.rename(columns=columns)
    return df

def read_json (fname):
    with open(fname, 'r') as fp:
        return json.load(fp)

def write_json (tax_table, fname):
    shutil.copyfile(fname, fname+'.bak')
    with open(fname, 'w') as fp:
        json.dump(tax_table, fp)

_manual_mutations_ = {
    "P04626[676-775,'YVMA',776-1255]" : 'MELAALCRWGLLLALLPPGAASTQVCTGTDMKLRLPASPETHLDMLRHLYQGCQVVQGNLELTYLPTNASLSFLQDIQEVQGYVLIAHNQVRQVPLQRLRIVRGTQLFEDNYALAVLDNGDPLNNTTPVTGASPGGLRELQLRSLTEILKGGVLIQRNPQLCYQDTILWKDIFHKNNQLALTLIDTNRSRACHPCSPMCKGSRCWGESSEDCQSLTRTVCAGGCARCKGPLPTDCCHEQCAAGCTGPKHSDCLACLHFNHSGICELHCPALVTYNTDTFESMPNPEGRYTFGASCVTACPYNYLSTDVGSCTLVCPLHNQEVTAEDGTQRCEKCSKPCARVCYGLGMEHLREVRAVTSANIQEFAGCKKIFGSLAFLPESFDGDPASNTAPLQPEQLQVFETLEEITGYLYISAWPDSLPDLSVFQNLQVIRGRILHNGAYSLTLQGLGISWLGLRSLRELGSGLALIHHNTHLCFVHTVPWDQLFRNPHQALLHTANRPEDECVGEGLACHQLCARGHCWGPGPTQCVNCSQFLRGQECVEECRVLQGLPREYVNARHCLPCHPECQPQNGSVTCFGPEADQCVACAHYKDPPFCVARCPSGVKPDLSYMPIWKFPDEEGACQPCPINCTHSCVDLDDKGCPAEQRASPLTSIISAVVGILLVVVLGVVFGILIKRRQQKIRKYTMRRLLQETELVEPLTPSGAMPNQAQMRILKETELRKVKVLGSGAFGTVYKGIWIPDGENVKIPVAIKVLRENTSPKANKEILDEAYVMAYVMAGVGSPYVSRLLGICLTSTVQLVTQLMPYGCLLDHVRENRGRLGSQDLLNWCMQIAKGMSYLEDVRLVHRDLAARNVLVKSPNHVKITDFGLARLLDIDETEYHADGGKVPIKWMALESILRRRFTHQSDVWSYGVTVWELMTFGAKPYDGIPAREIPDLLEKGERLPQPPICTIDVYMIMVKCWMIDSECRPRFRELVSEFSRMARDPQRFVVIQNEDLGPASPLDSTFYRSLLEDDDMGDLVDAEEYLVPQQGFFCPDPAPGAGGMVHHRHRSSSTRSGGGDLTLGLEPSEEEAPRSPLAPSEGAGSDVFDGDLGMGAAKGLQSLPTHDPSPLQRYSEDPTVPLPSETDGYVAPLTCSPQPEYVNQPDVRPQPPSPREGPLPAARPAGATLERPKTLSPGKNGVVKDVFAFGGAVENPEYLTPQGGAAPQPHPPPAFSPAFDNLYYWDQDPPERGAPPSTFKGTPTAENPEYLGLDVPV'}
        
def parse_uniprot_id (uid):
    if '[' not in uid:
        return uid, []
    
    base_uid, details = uid[:-1].split('[')
    mutations = []
    for det in details.split(','):
        if '-' in det:
            continue
        1
        frm = det[0]
        to = det[-1]
        pos = det[1:-1]
        if not pos.isnumeric():
            if uid in _manual_mutations_:
                return base_uid, ('manual', uid, _manual_mutations_[uid])
            else:
                raise Exception(uid)
        1
        mutations.append([frm, to, int(pos)-1])
    1
    return base_uid, mutations

def mod_ws_mutation (df):
    def mut2w (uid):
        if type(uid) is float and np.isnan(uid):
            return 1
        mutations = parse_uniprot_id(uid)[1]
        return (2+len(mutations)**0.75)
    df['weight'] *= df['uniprot_id'].apply(mut2w)
    

def apply_mutations (uid, seq, mutations):
    if len(mutations) == 0:
        return seq
    1
    if mutations[0] == 'manual':
        print(f'Applying manual mutation for uid: {mutations[1]}')
        return mutations[2]
    1
    seq = list(seq)
    for frm,to,pos in mutations:
        if seq[pos] == frm:
            seq[pos] = to
        else:
            raise Exception(f'Failed to apply mutations for {uid}')
    return ''.join(seq)
        
def build_uniprot_table (bdb, fname):
    table = read_json(fname)
    uniq_ids = bdb['uniprot_id'].unique()
    update_flag = False
    fail_lst = []
    for i,uid in enumerate(tqdm(uniq_ids, desc='Loading UniProt entries')):
        if type(uid) is float and np.isnan(uid):
            continue
        if table.get(uid) is not None:
            continue
        1
        try:
            print(uid)
            base_uid, mutations = parse_uniprot_id(uid)
            ret = requests.get(f'https://rest.uniprot.org/uniprotkb/{base_uid}.json')
            retval = ret.json()
            if retval.get('organism') is None:
                raise Exception(f'bad uniprot id: {uid}')
            1
            retval['sequence']['value'] = apply_mutations(uid, retval['sequence']['value'], mutations)
            table[uid] = retval
            update_flag = True
            if i % 500 == 0:
                write_json(table, fname)
        except Exception as e:
            print(e)
            fail_lst.append(uid)
    1
    table = {k:v for k,v in table.items() if 'organism' in v.keys()}
    if update_flag:
        print(f'Writing {fname}')
        write_json(table, fname)
    1
    if len(fail_lst) > 0:
        print('failed: ', fail_lst)
    1
    return table


def extract_floats (col):
    return col.str.extract('([-+]?\d*\.?\d+)', expand=False).astype(float)

def convert2floats (df, col_names):
    for col in col_names:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = extract_floats(df[col])
        else:
            print(f'Column {col} does not require float casting')
    return df

def mod_ws_apprx (df):
    col_names = ['Ki', 'Kd', 'IC50']
    apprx_set = []
    for col in col_names:
        apprx_ = df[col].str.contains('<') + df[col].str.contains('>')
        apprx = (apprx_ & ~apprx_.isna())
        apprx_set.append(apprx.to_numpy())

    ws_div = (np.sum(apprx_set, axis=0)>0)*3.16 + (np.sum(apprx_set, axis=0)==0)
    df['weight'] /= ws_div


def mod_ws_human (df, upt):
    def human_check (uid):
        return upt[uid]['organism']['scientificName'] == 'Homo sapiens'
        
    is_human = df['uniprot_id'].apply(human_check)
    df['weight'] *= (is_human*2) + ~is_human

def parse_apprx_value (val):
    if type(val) is float:
        return val
    val = val.strip()
    if val.startswith('>'):
        num = float(val[1:])
        num *= _cfg_['apprx_scale']
    elif val.startswith('<'):
        num = float(val[1:])
        num /= _cfg_['apprx_scale']
    else:
        num = float(val)
    return num

def convert_approx2floats (df, col_names):
    mod_ws_apprx(df)
    for col in col_names:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].apply(parse_apprx_value)
        else:
            print(f'Column {col} does not require approx parsing')
    return df


_virus_keys_ = set(['Pararnavirae',
                    'Orthornavirae',
                    'Ribozyviria',
                    'Shotokuvirae',
                    'Varidnaviria',
                    'Herpesvirales'])

_animal_keys_ = set(['Mammalia',
                     'Aves'])

_organism_keys_ = _virus_keys_ | _animal_keys_

def filter_organism_and_T (df, upt):
    mod_ws_human(df, upt)
    def animal_check (uid):
        lineage = set(upt[uid]['organism']['lineage'])
        return len(lineage & _animal_keys_) > 0
    
    def organism_check (uid):
        lineage = set(upt[uid]['organism']['lineage'])
        return len(lineage & _organism_keys_) > 0
    
    minval, maxval = _cfg_['pH_range']
    T_ok = ~((df['T'] < minval) | \
             (df['T'] > maxval))
    
    animal_ok = df['uniprot_id'].progress_apply(animal_check)
    organism_ok = df['uniprot_id'].progress_apply(organism_check)
    
    ok_ids = animal_ok | (T_ok & organism_ok)
    return filter_df('Organism/T', df, ~ok_ids)

def filter_df (name, df, rm_ids):
    begin_len = len(df)
    df_filtered = df[~rm_ids]
    end_len = len(df_filtered)
    n_removed = begin_len-end_len
    perc_removed = (n_removed/begin_len*100)
    print(f'{name} filter: {n_removed} of {begin_len} ({perc_removed:.3f}%) records removed')
    df_filtered.reset_index(drop=True, inplace=True)
    return df_filtered

def mod_ws_pH (df):
    minval, maxval = _cfg_['pH_range']
    good_ids = ~df['pH'].isna() & \
               (df['pH'] > minval) & \
               (df['pH'] < maxval) # not >=, border values are leaved but not buffed

    df['weight'][good_ids] *= 1.5


def filter_pH (df):
    minval, maxval = _cfg_['pH_range']
    rm_ids = (df['pH'] < minval) | \
             (df['pH'] > maxval)
    df = filter_df('pH', df, rm_ids)
    mod_ws_pH(df)
    return df

#def filter_T (df):
#    minval, maxval = __cfg__['T_range']
#    rm_ids = (df['T'] < minval) | \
#             (df['T'] > maxval)
#    return filter_df('T', df, rm_ids)

_inactive_ids_ = ['F5H9H0', 'D0IQW9', 'J0VBT9', 'A0A3S3XYR9', 'A0A8B5A132',
                  'T2NA71', 'A0A024R3I6', 'A0A068N674', 'B2V8E3', 'T1Y9P4',
                  'A0A510WMC0', '50001261',
                  # bad mutation codes:
                  'P08581[974-1390,A1209G,V1290L]',
                  "Q9BHJ5[3-643,Y342H]",
                  "Q9BHJ5[3-643,D247A]"]

def filter_uniprot (df):
    no_ids = df['uniprot_id'].isna()
    complex_ids = df['uniprot_id'].str.contains(' ')
    bad_match = '|'.join([re.escape(uid) for uid in _inactive_ids_])
    inactive_ids = df['uniprot_id'].str.contains(bad_match)
    rm_ids = no_ids | complex_ids | inactive_ids
    return filter_df('Has UniProt id', df, rm_ids)

def filter_one_of_Kd_Ki_IC50 (df):
    rm_ids = df['Kd'].isna() & \
             df['Ki'].isna() & \
             df['IC50'].isna()
    return filter_df('No Kd or Ki or IC50', df, rm_ids)

def weighted_percentile (data, weights, perc):
    if len(data) == 0:
        return np.nan
    ix = np.argsort(data)
    data = data[ix]
    weights = weights[ix]
    cdf = (np.nancumsum(weights) - 0.5*weights) / np.sum(weights)
    return np.interp(perc/100, cdf, data)

def aggregate_constants (df):
    groups = df.groupby(['smiles', 'uniprot_id']).groups

    agg = []
    for (inchi, uniprot_id), ids in tqdm(groups.items()):
        grp = df.iloc[ids]
        kd_ki = agg_kd_ki(grp)
        ic50 = agg_ic50(grp)
        w = agg_ws(grp)
        agg.append([inchi, uniprot_id, kd_ki, ic50, w])
        
    df_agg = pd.DataFrame(agg,
                          columns = ['smiles',
                                     'uniprot_id',
                                     'Ki',
                                     'IC50',
                                     'weight'])
    return df_agg

def mod_ws_both_constants (df):
    has_ki_kd = (~df['Kd'].isna()) | \
                (~df['Ki'].isna())
    has_ic50 = ~df['IC50'].isna()
    has_both = has_ki_kd & has_ic50
    df['weight'][has_both] *= 3.16
    

def clip_constants (df):
    ki_min, ki_max =_cfg_['Kd/Ki_range']
    ic_min, ic_max =_cfg_['IC50_range']

    outlier_ids = ((df['Ki'] < ki_min) & \
                   (df['Ki'] > ki_max) & \
                   (df['Kd'] < ki_min) & \
                   (df['Kd'] > ki_max) & \
                   (df['IC50'] < ic_min) & \
                   (df['IC50'] > ic_max))

    df['weight'][outlier_ids] /= 3.16
    df['Kd'] = np.clip(df['Kd'], ki_min, ki_max)
    df['Ki'] = np.clip(df['Ki'], ki_min, ki_max)
    df['IC50'] = np.clip(df['IC50'], ki_min, ki_max)
        
def agg_ic50 (grp):
    good_ids = ~grp['IC50'].isna()
    vals = grp['IC50'][good_ids].to_numpy()
    ws = grp['weight'][good_ids].to_numpy()
    agg_val = weighted_percentile(vals, ws, _cfg_['IC50_percentile'])
    return agg_val

def agg_ws (grp):
    ws = grp['weight'].to_numpy()
    return np.mean(ws) * (1+np.log10(len(ws)))
    
def agg_kd_ki (grp):
    good_ki_ids = ~grp['Ki'].isna()
    good_kd_ids = ~grp['Kd'].isna()
    ki_vals = grp['Ki'][good_ki_ids].to_numpy()
    kd_vals = grp['Kd'][good_kd_ids].to_numpy()
    ki_ws = grp['weight'][good_ki_ids].to_numpy()
    kd_ws = grp['weight'][good_kd_ids].to_numpy()
    vals = np.hstack((ki_vals, kd_vals))
    ws = np.hstack((ki_ws, kd_ws))
    agg_val = weighted_percentile(vals, ws, _cfg_['Kd/Ki_percentile'])
    return agg_val

'''
def agg_kd_ki_old (grp):
    kd_ki = np.hstack((grp['Ki'], grp['Kd']))
    kd_ki_clip = np.clip(kd_ki, *_cfg_['Kd/Ki_range'])
    agg_val = np.nanpercentile(kd_ki_clip, _cfg_['Kd/Ki_percentile'])
    return agg_val

def agg_ic50_old (grp):
    ic50 = grp['IC50']
    ic50_clip = np.clip(ic50, *_cfg_['IC50_range'])
    agg_val = np.nanpercentile(ic50_clip, _cfg_['IC50_percentile'])
    return agg_val
'''
                          
def filter_convert_smiles (df):
    def convert_fn (sm):
        try:
            can_sm = Chem.MolToSmiles(
                Chem.MolFromSmiles(sm),
                canonical=True,
                isomericSmiles=True)
            if can_sm is None:
                raise Exception
            if len(can_sm) > 300: # 512 limit in chemformer - handicap for augment
                raise Exception
        except:
            #print(f'bad smiles: {sm}')
            can_sm = np.nan
        
        return can_sm
    
    df['smiles'] = df['smiles'].progress_apply(convert_fn)
    
    rm_ids = df['smiles'].isna()
    return filter_df('SMILES converter', df, rm_ids)

def set_protein (df, upt):
    proteins = [upt[uid]['sequence']['value'] \
                for uid in df['uniprot_id']]
    df['protein'] = proteins
    return df

# protein impact weights
def read_protein_classes (path='UniProtID_protein_class.csv'):
    cls_df = pd.read_csv(path)
    cls_table = {}
    cls_dict = {}
    for i,row in cls_df.iterrows():
        uid = row['uniprot_id']
        classes = row['protein_class'].split(',')
        cls_dict[uid] = classes
        for cls in classes:
            if cls_table.get(cls) is None:
                cls_table[cls] = []
            cls_table[cls].append(uid)
    return cls_table, cls_dict

def make_prot_cls_ws (cls_table, cls_dict):
    ws_table = {}
    for uid in cls_dict:
        uid_w = 1
        for cls in cls_dict[uid]:
            cls_w = len(cls_dict)/len(cls_table[cls])
            uid_w *= cls_w
        ws_table[uid] = np.log10(uid_w)**0.5
    return ws_table

def make_prot_uid_ws (bdb):
    uid_counter = Counter(bdb['uniprot_id'])
    ws_table = {}
    for uid,n_occur in uid_counter.items():
        w = 2/(1+np.log10(n_occur))
        ws_table[uid] = w
    return ws_table

def uid2w (uid, cls_ws, uid_ws):
    return (cls_ws.get(uid, 1) * uid_ws[uid])**0.5


def mod_ws_prot (df_agg):
    cls_table, cls_dict = read_protein_classes()
    cls_ws = make_prot_cls_ws(cls_table, cls_dict)
    uid_ws = make_prot_uid_ws(df_agg)
    ws = df_agg['uniprot_id'].apply(lambda uid: uid2w(uid, cls_ws, uid_ws))
    df_agg['weight'] *= ws
    df_agg['weight'] /=df_agg['weight'].mean()


# protein mol wt
def filter_weight (df, upt):
    minval, maxval = _cfg_['molwt_range']
    
    def weight_fn (uid):
        wt = upt[uid]['sequence']['molWeight']
        return minval <= wt <= maxval

    weight_ids = df['uniprot_id'].progress_apply(weight_fn)
    return filter_df('MolWeight', df, ~weight_ids)

def filter_len (df, upt):
    minval, maxval = _cfg_['protein_len_range']
    
    def len_fn (uid):
        ln = upt[uid]['sequence']['length']
        return minval <= ln <= maxval

    lens_ids = df['uniprot_id'].progress_apply(len_fn)
    return filter_df('Protein len', df, ~lens_ids)

def convert2log (val):
    return 9-np.log10(val)

def run (src, tgt):
    global bdb
    bdb = read_bindingdb(src)
    fill_missing_uniprot_ids(bdb)
    mod_ws_mutation(bdb)
    bdb = filter_one_of_Kd_Ki_IC50(bdb)
    bdb = convert_approx2floats(bdb, col_names=['Ki', 'Kd', 'IC50'])
    bdb = convert2floats(bdb, col_names=['Ki', 'Kd', 'IC50', 'T', 'pH'])
    clip_constants(bdb)
    mod_ws_both_constants(bdb)
    #upt = read_uniprot('uniprot.csv')
    upt = build_uniprot_table(bdb, 'uniprot_table.json')
    bdb = filter_uniprot(bdb)
    
    bdb = filter_pH(bdb)
    bdb = filter_organism_and_T(bdb, upt)
    bdb = filter_convert_smiles(bdb)

    global df_agg
    df_agg = aggregate_constants(bdb)
    df_agg = filter_weight(df_agg, upt)
    df_agg = filter_len(df_agg, upt)
    
    df_agg['pKi'] = df_agg['Ki'].apply(convert2log)
    df_agg['pIC50'] = df_agg['IC50'].apply(convert2log)
    
    df_agg = set_protein(df_agg, upt)
    mod_ws_prot(df_agg)
    df_agg['weight'] = df_agg['weight'].apply(lambda x: x**0.5)

    train_ids, test_ids = cold_drug_target_split(df_agg, 'smiles', 'protein',
                                                 test_part=0.05)
    df_agg.iloc[train_ids].to_csv(tgt+'.csv', index=False)
    df_agg.iloc[test_ids].to_csv(tgt+'_test.csv', index=False)
    #train_test_split(df_agg, tgt, 0.01)
    
    #df_agg.to_csv(tgt,
    #              index=False,
    #              columns=['smiles', 'protein', 'pKi', 'pIC50', 'weight'])
    #print(f'Result file `{tgt}` written')

# command to make diff
# head -n 1 BindingDB_All_202404.tsv > BindingDB_All_202404_diff.tsv && cat diff.tsv | grep -E '^>' | sed s/^\>\ // >> BindingDB_All_202404_diff.tsv

def plot_weights (df):
    plt.hist(df.weight, bins=200)
    plt.axvline(df.weight.mean(),
                color='green',
                linestyle='dashed',
                linewidth=1.5,
                alpha=0.75,
                label=f'Mean weight: {df.weight.mean():.2f}')
    plt.axvline(df.weight.min(), 0, 0.5,
                color='red',
                linestyle='dashed',
                linewidth=1.5,
                alpha=0.5,
                label=f'Min  weight: {df.weight.min():.2f}')
    plt.axvline(df.weight.max(), 0, 0.5,
                color='red',
                linestyle='dashed',
                linewidth=1.5,
                alpha=0.5,
                label=f'Max weight: {df.weight.max():.2f}')
    plt.plot([], color='white', label=f'Std.dev: {np.std(df.weight):.2f}')
    plt.legend(loc='upper right', fontsize=12)
    plt.title('Sample weight distribution')
    plt.minorticks_on()
    plt.xlabel('Sample weight')
    plt.ylabel('Number of samples')
    plt.savefig("weights_dist.png")
    plt.close()

def make_float (s): return float(str(s).strip(' <>'))
    
def plot_raw_stats (df, key):
    if key == 'IC50':
        vals_raw = df[~df[key].isna()][key]
    elif key == 'Ki':
        vals_raw = df[~df[key].isna()][key].tolist() + df[~df['Kd'].isna()]['Kd'].tolist()
    else: 1/0
    vals = np.array([convert2log(make_float(s)) for s in vals_raw])
    vals_apprx = np.array([convert2log(make_float(s)) for s in vals_raw if ('>' in str(s) or '<' in str(s))])
    vals[vals==np.inf] = 15
    vals_apprx[vals_apprx==np.inf] = 15
    plt.plot([], color='white', label='200 bins')
    h,bins,bars = plt.hist(vals, bins=200, label='All values')
    plt.hist(vals_apprx, bins=bins, label='<> values')
    perc_min = np.percentile(vals, 0.5)
    perc_max = np.percentile(vals, 99.5)
    plt.axvline(perc_min, 0, 1,
                color='red',
                linestyle='dashed',
                linewidth=1.5,
                alpha=0.5,
                label=f'99% percentile\n inside: {perc_min:.1f}-{perc_max:.1f}')
    plt.axvline(perc_max, 0, 1,
                color='red',
                linestyle='dashed',
                linewidth=1.5,
                alpha=0.5)
    1
    plt.legend()
    plt.xlim(0,12)
    plt.xticks(list(range(1,12)))
    plt.xlabel(f'p{key} value')
    plt.ylabel('Samples count')
    plt.title(f'Raw values distribution (p{key})')
    plt.savefig(f"plot_p{key}.png")
    plt.close()
    
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        src = 'BindingDB_All_202406.tsv'
        tgt = 'bindingdb'
    else:
        src = sys.argv[1]
        tgt = sys.argv[2]

    run(src, tgt)
    
