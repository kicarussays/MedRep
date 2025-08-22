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
parser.add_argument('--hospital', type=str, default='ehrshot')
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
spath = os.path.join(abspath, 'usedata/{hosp}/')
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
allrecords = mpd.read_csv(os.path.join(spath.format(hosp=args.hospital), 'allrecords_divided.csv'))
person = mpd.read_csv(os.path.join(dpath, 'person.csv'))
genders = person[['person_id', 'gender_concept_id']]
genders['gender_source_value'] = person['gender_concept_id'].apply(lambda x: 'M' if x == 8507 else 'F')
genders = genders[['person_id', 'gender_source_value']].set_index('person_id')

visit = mpd.read_csv(os.path.join(dpath, 'visit_occurrence.csv'))
visit_label = mpd.read_csv(os.path.join(spath.format(hosp=args.hospital), 'visit_label.csv'))
visit_label['label'] = visit_label['label'].apply(lambda x: 1 if x else 0)
phe = mpd.read_csv(os.path.join(spath.format(hosp=args.hospital), 'phenotypes.csv'))
phe['label'] = phe.iloc[:, 2:].values.tolist()
phe['label_type'] = 'Pheno'
visit_label = mpd.concat([visit_label, phe[visit_label.columns]]).reset_index(drop=True)
visit_label['prediction_timepoint'] = visit_label['prediction_timepoint'].apply(mpd.to_datetime)
print('Done')


# Screen data
allpid = visit_label['person_id'].unique()
allrecords = allrecords[allrecords['person_id'].isin(allpid)]
allrecords['record_datetime'] = allrecords['record_datetime'].apply(mpd.to_datetime)
allrecords.reset_index(drop=True, inplace=True)

# Tokenization
outcomes = visit_label['label_type'].unique()
os.makedirs(os.path.join(tpath, 'Finetuning'), exist_ok=True)
for outcome in ['Pheno', ]:
# for outcome in outcomes:
    print(f'\n\n{outcome} tokenizing...')
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

    _allrecords = userecords[userecords['person_id'].isin(allpid)]
    dpids = np.array_split(_allrecords['person_id'].unique(), 2) # raise the number if dataset is too large

    # for hosp in ('mimic', ):
    for hosp in ('mimic', 'ehrshot'):
        if args.hospital == 'mimic' and hosp == 'ehrshot': continue
        savepath = os.path.join(tpath, 'Finetuning', f'{outcome}_{hosp}.pkl')
        vocab = torch.load(os.path.join(spath.format(hosp=hosp), 'vocab.pt'))
        tokenizer = EHRTokenizer(vocabulary=vocab, config=tokenizer_config)

        all_dpid_records = []
        all_dpid_labels = []
        for n, dpid in enumerate(dpids):
            dpid_records = _allrecords[_allrecords['person_id'].isin(dpid)]
            gender_info = genders.loc[dpid].values.reshape(-1)
            features, labels = featurization(dpid_records, gender_info, outcome)
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

        picklesave((tokenized, labels), savepath)
        