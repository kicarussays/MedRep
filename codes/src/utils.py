import torch
import numpy as np
import pandas as pd
import random
import pickle
from sklearn.metrics import roc_curve
from tqdm import tqdm


def seedset(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def picklesave(file, path):
    with open(path, 'wb') as f:
        pickle.dump(file, f, pickle.HIGHEST_PROTOCOL)


def pickleload(path):
    with open(path, 'rb') as f:
        tmp = pickle.load(f)
    
    return tmp


class DotDict(dict):
    def __getattr__(self, attr):
        value = self.get(attr)
        if isinstance(value, dict):
            return DotDict(value)
        return value


def featurization(records, gender_info, OUTCOME=None):
    use_id = 'divided_person_id' if OUTCOME is None else 'person_id'
    use_cols = ['concept_id', 'domain', 'age', 'visit_rank', 'record_rank']

    _features = records.groupby([use_id]).apply(
            lambda x: x[use_cols].values)

    features = {
        'concept': [],
        'domain': [],
        'age': [],
        'segment': [],
        'record_rank': [],
    }
    for n, i in tqdm(enumerate(_features)):
        features['concept'].append([gender_info[n]] + i[:, 0].tolist())
        features['domain'].append([0] + i[:, 1].astype(int).tolist())
        features['age'].append([i[0, 2]] + i[:, 2].astype(int).tolist())
        features['segment'].append([0] + i[:, 3].astype(int).tolist())
        features['record_rank'].append([0] + i[:, 4].astype(int).tolist())

    if OUTCOME is None:
        return features
    else:
        label = list(records.groupby([use_id]).apply(lambda x: list(x['label'])[0]))
        return features, label


def r2(v):
    return "{:.2f}".format(round(v, 2))
def r3(v):
    return "{:.3f}".format(round(v, 3))
def r4(v):
    return "{:.4f}".format(round(v, 4))


def pred_calculation(y, y_hat):
    fpr, tpr, thresholds = roc_curve(y, y_hat)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix] 
    y_pred = (y_hat >= best_thresh).astype(bool)
    return y_pred


def finish_check(path):
    f = open(path, 'r')
    lines = f.readlines()
    flag = False
    for line in lines:
        line = line.strip()  
        if 'Early stopping activated' in line or 'Epoch [100/100]' in line:
            flag = True
            break
    return flag