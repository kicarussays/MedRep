"""
    Tokenize for RETAIN, BEHRT, BEHRT-DE, MedBERT
"""

import os
from tqdm import tqdm
import json
import ray
import modin.pandas as mpd
import numpy as np
import pandas as pd
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_cpus', type=int, default=64)
parser.add_argument('--hospital', type=str, default='mimic')
parser.add_argument('--seq-length', type=int, default=2048)
args = parser.parse_args()


# Set absolute path
import sys
from pathlib import Path
abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)
from src.tokenizer import EHRTokenizer
from src.utils import featurization, picklesave
from src.vars import tokenizer_config

abspath = str(Path(__file__).resolve().parent.parent.parent)
dpath = os.path.join(abspath, f'data/{args.hospital}/')
spath = os.path.join(abspath, f'usedata/{args.hospital}/')
tpath = os.path.join(abspath, f'usedata/{args.hospital}/tokens/')
os.makedirs(tpath, exist_ok=True)


# Turn on ray
os.makedirs(os.path.join(abspath, "trashbin/ray_spill"), exist_ok=True)
ray.init(
    num_cpus=args.num_cpus,
    _system_config={
        "object_spilling_config": json.dumps(
            {"type": "filesystem", "params": {
                "directory_path": os.path.join(abspath, "trashbin/ray_spill")
            }}
        )
    },
    object_store_memory=int(200*1024*1024*1024)
)


# Data load
print('Data load...')
allrecords = mpd.read_csv(os.path.join(spath, 'allrecords_divided.csv'))
train_id = pd.read_csv(os.path.join(spath, 'train_id.csv'))
valid_id = pd.read_csv(os.path.join(spath, 'valid_id.csv'))
test_id = pd.read_csv(os.path.join(spath, 'test_id.csv'))
person = mpd.read_csv(os.path.join(dpath, 'person.csv'))
visit = mpd.read_csv(os.path.join(dpath, 'visit_occurrence.csv'))
print('Done')


# Screen person_id / Label plos
allpid = np.sort(np.concatenate([
    train_id, valid_id, test_id
]).squeeze())
allrecords = allrecords[allrecords['person_id'].isin(allpid)]

person = person[person['person_id'].isin(allpid)]
visit = visit[visit['person_id'].isin(allpid)]
visit['date_diff'] = (mpd.to_datetime(visit['visit_end_date']) - \
                      mpd.to_datetime(visit['visit_start_date'])).dt.days
plos_person = visit[visit['date_diff'] >= 7]['person_id'].unique()


# Tokenization - for PRETRAINING (train and valid)
print('Tokenizing for pretraining...')
vocab = torch.load(os.path.join(spath, 'vocab.pt'))
tokenizer = EHRTokenizer(vocabulary=vocab, config=tokenizer_config)
tokenizer.freeze_vocabulary()

allpid_dict = {
    'train': train_id,
    'valid': valid_id,
}


for k, v in allpid_dict.items():
    print(f'###### {k} ######')
    _allrecords = allrecords[allrecords['person_id'].isin(v['person_id'])]
    dpids = np.array_split(_allrecords['divided_person_id'].unique(), 30)

    all_dpid_records = []
    all_plos_label = []
    for n, dpid in enumerate(dpids):
        print(f'dpid idx: {n+1} featurizing...')
        dpid_records = _allrecords[_allrecords['divided_person_id'].isin(dpid)]
        features = featurization(dpid_records)
        print('Done')
        print('Tokenizing...')
        tokenized = tokenizer(features)
        all_dpid_records.append(tokenized)

        _pids = dpid_records['divided_person_id'].drop_duplicates(keep='first') 
        plos_label = [int(i.split('_')[0]) in plos_person for i in tqdm(_pids)] # for medbert
        all_plos_label.append(plos_label)
        print('Done')
        
    tokenized = {}
    for j in all_dpid_records[0].keys():
        tokenized[j] = torch.cat([d[j] for d in all_dpid_records])
    plos_label = np.concatenate(all_plos_label).tolist()

    picklesave(tokenized, os.path.join(tpath, f'Pretraining_tokens_{k}.pkl'))
    picklesave(plos_label, os.path.join(tpath, f'Pretraining_plos_{k}.pkl'))
print('Done')
