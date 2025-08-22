import os
from tqdm import tqdm
import json
import ray
import modin.pandas as mpd
import numpy as np
import pandas as pd
import datetime
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_cpus', type=int, default=96)
parser.add_argument('--hospital', type=str, default='snuh')
parser.add_argument('--seq-length', type=int, default=2048)
args = parser.parse_args()


# Set absolute path
import sys
from pathlib import Path
abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)
from src.tokenizer import EHRTokenizer
from src.utils import DotDict, featurization, picklesave
from src.vars import tokenizer_config, outcome_prediction_point

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
allrecords = mpd.read_csv(os.path.join(spath, 'allrecords.csv'))
train_id = pd.read_csv(os.path.join(spath, 'train_id.csv'))
valid_id = pd.read_csv(os.path.join(spath, 'valid_id.csv'))
test_id = pd.read_csv(os.path.join(spath, 'test_id.csv'))
person = mpd.read_csv(os.path.join(dpath, 'person.csv'))
visit = mpd.read_csv(os.path.join(dpath, 'visit_occurrence.csv'))
visit.columns = [i.lower() for i in visit.columns]
print('Done')

concepts = mpd.read_csv(os.path.join(abspath, 'usedata/representation/concept_idx_full.csv'))
concepts['concept_id'] = concepts['concept_id'].astype(str)
allrecords['concept_id'] = allrecords['concept_id'].astype(str)
allrecords = allrecords[allrecords['concept_id'].isin(concepts['concept_id'])]
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


# Merge allrecords and visit
print('Merge allrecords and visit...')
allrecords = allrecords[['person_id', 'concept_id', 'record_datetime', 'domain']]
visit = visit[['person_id', 'visit_concept_id', 'visit_start_datetime']].drop_duplicates()
visit['domain'] = 'visit'
visit.columns = ['person_id', 'concept_id', 'record_datetime', 'domain']
visit = visit[['person_id', 'record_datetime', 'domain', 'concept_id']]
allrecords = mpd.concat([allrecords, visit]).reset_index(drop=True)
allrecords['record_datetime'] = mpd.to_datetime(allrecords['record_datetime'])
print('Done')


# Age setting
print('Age setting...')
if args.hospital == 'mimic':
    patients = mpd.read_csv(os.path.join(dpath, 'patients.csv'))
    person['subject_id'] = person['trace_id'].apply(lambda x: x.split(':')[1][:-1]).astype(int)
    person = mpd.merge(
        person, patients[['subject_id', 'anchor_age']],
        on='subject_id', how='left'
    )
    person['year_of_birth'] = person['year_of_birth'] - person['anchor_age']

allrecords = mpd.merge(
    allrecords,
    person[['person_id', 'year_of_birth']],
    on=['person_id'], how='left'
)
allrecords['record_year'] = mpd.to_datetime(allrecords['record_datetime']).dt.year
allrecords['age'] = allrecords['record_year'] - allrecords['year_of_birth']
allrecords['concept_id'] = allrecords['concept_id'].astype(str)
allrecords = allrecords[['person_id', 'concept_id', 'record_datetime', 'domain', 'age']]
print('Done')


# Assign visit and domain index
print('Assign visit and domain index...')
ar_visit = allrecords[allrecords['domain'] == 'visit'][['person_id', 'record_datetime']].sort_values(
    ['person_id', 'record_datetime']
)
ar_visit['record_date'] = mpd.to_datetime(ar_visit['record_datetime']).dt.date
ar_visit['row'] = (ar_visit.groupby(['person_id', 'record_date']).cumcount()+1).iloc[:, 0] # modin.pandas bug
ar_visit = ar_visit[ar_visit['row'] == 1].drop(columns=['row'])
ar_visit['visit_rank'] = ar_visit.groupby('person_id')['record_date'].rank(method='dense')
allrecords.loc[ar_visit.index, 'visit_rank'] = list(ar_visit['visit_rank'].astype(int))
allrecords['visit_rank'] = allrecords.groupby(
    'person_id')['visit_rank'].apply(lambda x: x.bfill().ffill()).values
allrecords['visit_rank'] = allrecords['visit_rank'].fillna(1)

domain_idx = ['special_token', 'cond', 'drug', 'meas', 'proc', 'visit']
allrecords['domain'] = allrecords['domain'].apply(lambda x: domain_idx.index(x))
allrecords['concept_id'] = allrecords['concept_id'].astype(str)
allrecords = allrecords[allrecords['domain'] != 5] # Omit visit records
print('Done')


# Dividing long sequences
print('Dividing long sequences')
allrecords['divided_idx'] = (allrecords.groupby(
    ['person_id']).cumcount()+1).iloc[:, 0] // args.seq_length
allrecords['divided_person_id'] = allrecords['person_id'].astype(str) + '_' + \
    allrecords['divided_idx'].astype(str)
allrecords['visit_rank'] = allrecords.groupby(
    ['divided_person_id'])['visit_rank'].rank(method='dense')
allrecords['record_rank'] = allrecords.groupby(
    ['divided_person_id', 'visit_rank'])['record_datetime'].rank(method='dense')
print('Saving...')
allrecords['age'] = allrecords['age'].apply(lambda x: x if x >= 18 else 18)
allrecords.to_csv(os.path.join(spath, 'allrecords_divided.csv'), index=None)
print('Done')


# Save vocab
print('Save vocab...')
vocab = allrecords['concept_id'].unique().astype(str)
vocab = np.concatenate([
    np.array(['[PAD]', '[MASK]', '[UNK]', '[CLS]', '[SEP]', 'M', 'F']), vocab])
vocab = {str(v): n for n, v in enumerate(vocab)}
torch.save(vocab, os.path.join(spath, 'vocab.pt'))
with open(os.path.join(spath, 'vocab.json'), 'w') as f:
    json.dump(vocab, f)
print('Done')



