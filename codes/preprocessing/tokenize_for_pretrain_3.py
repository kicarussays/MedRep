"""
    Tokenize for Ethos
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
parser.add_argument('--num_cpus', type=int, default=96)
parser.add_argument('--hospital', type=str, default='mimic')
parser.add_argument('--seq-length', type=int, default=2048)
args = parser.parse_args()


# Set absolute path
import sys
from pathlib import Path
abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)
from src.tokenizer import EHRTokenizerForEthos
from src.utils import picklesave

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
tokenizer = EHRTokenizerForEthos(vocabulary=vocab)

allpid_dict = {
    'train': train_id,
    'valid': valid_id,
}

for k, v in allpid_dict.items():
    print(f'{k} tokenizing...')
    _allrecords = allrecords[allrecords['person_id'].isin(v['person_id'])]
    allrecords_for_tokenize = {
        'concept': list(_allrecords['concept_id']),
        'domain': list(_allrecords['domain']),
        'age': list(_allrecords['age']),
        'segment': list(_allrecords['visit_rank']),
        'record_rank': list(_allrecords['record_rank']),
    }
    tokenized = tokenizer(allrecords_for_tokenize)
    tokenized = {k: torch.Tensor(np.array(v)).type(torch.int32) for k, v in tokenized.items()}
    picklesave(tokenized, os.path.join(tpath, f'Pretraining_tokens_ethos_{k}.pkl'))

print('Done')
