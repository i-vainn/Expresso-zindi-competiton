import pandas as pd
import numpy as np
import yaml
import warnings
warnings.filterwarnings('ignore')

def read_data(path='../data/', who=['train', 'test', 'samplesubmission']):
    data = []
    for file in who:
        data.append(pd.read_csv(path + file + '.csv'))
    return data

def make_map(s):
    s = str(s)
    if s[:-1].isalpha():
        return np.NaN
    if s.endswith('GB'):
        return eval(s[:-2] + ' * 1000')
    if s.endswith('MB'):
        return int(s[:-2])
    return int(s)

def make_time_map(s):
    s = str(s)
    if s.endswith('d'):
        return float(s[:-1])
    elif s.endswith('H'):
        return float(s[:-1]) / 24
    return np.NaN

def preprocess(data):
    with open('../tpmap.yaml') as f:
        mapper = yaml.load(f)
        
    data = pd.concat([
        data.drop('TOP_PACK', axis=1),
        pd.DataFrame(data.TOP_PACK.fillna('NAN').map(mapper).to_list())],
        axis=1
    )

    data['size'] = data['size'].map(make_map)
    data['duration'] = data['duration'].map(make_time_map)
    data['F'] = data['F'].astype('float')
    
    return data

def handle_nans(*tables):
    main = tables[0]
    for col in main.columns:
        if main[col].isna().sum() == 0:
            continue

        filler = 'NAN'
        if main[col].dtype != 'object':
            for sub in tables:
                sub[col + '_isna'] = sub[col].isna().astype('int')
            
            filler = np.median([sub[col][sub[col].notna()] for sub in tables])

        for sub in tables:
            sub[col] = sub[col].fillna(filler)
    return tables

def make_submission(prediction, filename, path='../data/submissions'):
    sub['CHURN'] = prediction
    sub.to_csv('{}/{}.csv'.format(path, filename), index=False)