import os
import ray
import json
import modin.pandas as mpd
import pandas as pd
import numpy as np
import warnings
import argparse
import torch
import argparse

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=Warning)

parser = argparse.ArgumentParser()
parser.add_argument('--num_cpus', type=int, default=64)
parser.add_argument('--hospital', type=str, default='mimic')
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
allrecords = mpd.read_csv(os.path.join(spath, 'allrecords_divided.csv'))
train_id = pd.read_csv(os.path.join(spath, 'train_id.csv'))
valid_id = pd.read_csv(os.path.join(spath, 'valid_id.csv'))
test_id = pd.read_csv(os.path.join(spath, 'test_id.csv'))
person = mpd.read_csv(os.path.join(dpath, 'person.csv'))
visit = mpd.read_csv(os.path.join(dpath, 'visit_occurrence.csv'))
visit_label = mpd.read_csv(os.path.join(spath, 'visit_label.csv'))
visit_label['prediction_timepoint'] = mpd.to_datetime(visit_label['prediction_timepoint'])
print('Done')


# Screen data
allpid = visit_label['person_id'].unique()
allrecords = allrecords[allrecords['person_id'].isin(allpid)]
allrecords['record_datetime'] = mpd.to_datetime(allrecords['record_datetime'])

allpid_dict = {
    'train': train_id,
    'valid': valid_id,
    'test': test_id,
}
if args.hospital == 'ehrshot':
    allpid_dict = {
        'test': pd.concat([train_id, valid_id, test_id])
    }


# Tokenization
os.makedirs(os.path.join(tpath, 'Finetuning'), exist_ok=True)
for outcome in ('MT', 'LLOS', 'RA'):
    print(f'{outcome} tokenizing...')
    uselabel = visit_label[visit_label['label_type'] == outcome]
    userecords = mpd.merge(
        allrecords, uselabel,
        on=['person_id'], how='left'
    )
    userecords = userecords[
        (userecords['record_datetime'] < userecords['prediction_timepoint'])
    ]
    userecords = userecords.sort_values(
        ['person_id', 'record_datetime'], ascending=[True, False])
    userecords['row'] = userecords.groupby(['person_id']).cumcount()+1
    userecords = userecords[userecords['row'] <= 2048].sort_values([
        'person_id', 'record_datetime'
    ])

    for k, v in allpid_dict.items():
        print(f'{k} featurizing...')
        _allrecords = userecords[userecords['person_id'].isin(v['person_id'])]
        dpids = np.array_split(_allrecords['person_id'].unique(), 2) # raise the number if dataset is too large

        if args.hospital == 'mimic':
            from_vocab = ['mimic']
        else:
            from_vocab = ['mimic', args.hospital]
        for h in from_vocab:
            spath = os.path.join(abspath, f'usedata/{h}/')
            vocab = torch.load(os.path.join(spath, 'vocab.pt'))
            tokenizer = EHRTokenizer(vocabulary=vocab, config=tokenizer_config)

            all_dpid_records = []
            all_dpid_labels = []
            for n, dpid in enumerate(dpids):
                dpid_records = _allrecords[_allrecords['person_id'].isin(dpid)]
                features, labels = featurization(dpid_records, outcome)
                print('Done')
                print('Tokenizing...')

                tokenizer.freeze_vocabulary()
                tokenized = tokenizer(features)
                all_dpid_records.append(tokenized)
                all_dpid_labels.append(labels)
        
            tokenized = {}
            for j in all_dpid_records[0].keys():
                tokenized[j] = torch.cat([d[j] for d in all_dpid_records])
            labels = np.concatenate(all_dpid_labels).tolist()

            picklesave(
                (tokenized, labels), 
                os.path.join(
                    tpath, 
                    'Finetuning', 
                    f'{outcome}_{args.hospital}_from_vocab_{h}_{k}.pkl'))

