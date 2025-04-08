import os
from tqdm import tqdm
import bisect
import ray
import json
import modin.pandas as mpd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import warnings
import re


warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--num_cpus', type=int, default=112)
parser.add_argument('--hospital', type=str)
args = parser.parse_args()

import sys
from pathlib import Path
abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)
from src.utils import picklesave

abspath = str(Path(__file__).resolve().parent.parent.parent)
dpath = os.path.join(abspath, f'data/{args.hospital}/')
spath = os.path.join(abspath, f'usedata/{args.hospital}/')
cpath = os.path.join(abspath, 'data/concepts')
os.makedirs(spath, exist_ok=True)


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


# Use certified concept ID
co = mpd.read_csv(os.path.join(cpath, 'CONCEPT.csv'), delimiter='\t')
co = co[co['domain_id'].isin(['Condition', 'Drug', 'Measurement', 'Procedure'])]
person = mpd.read_csv(os.path.join(dpath, 'person.csv'))
person.columns = [i.lower() for i in person.columns]
if args.hospital == 'mimic':
    patients = mpd.read_csv(os.path.join(dpath, 'patients.csv'))
    person['subject_id'] = person['trace_id'].apply(lambda x: x.split(':')[1][:-1]).astype(int)
    person = mpd.merge(
        person, patients[['subject_id', 'anchor_age']],
        on='subject_id', how='left'
    )
    person['year_of_birth'] = person['year_of_birth'] - person['anchor_age']
    
    
############ cond
############ cond
print('Cond processing...')
cond = mpd.read_csv(os.path.join(dpath, 'condition_occurrence.csv'))
cond.columns = [i.lower() for i in cond.columns]
cond['condition_concept_id'] = cond['condition_concept_id'].astype(int)
cond = cond[(cond['condition_concept_id'] != 0) & (cond['condition_concept_id'].isin(co['concept_id']))]
cond['condition_start_datetime'] = cond['condition_start_datetime'].fillna(cond['condition_start_date'])
cond = cond[
    (cond['condition_start_datetime'] >= '1900-01-01') & 
    (cond['condition_start_datetime'] <= '2260-12-31')]
cond = mpd.merge(
    cond,
    person[['person_id', 'year_of_birth']],
    on='person_id', how='left'
)
cond = cond[
    cond['condition_start_datetime'].apply(lambda x: x[:4]).astype(int) - cond['year_of_birth'] >= 18]
cond = cond[['person_id', 'condition_concept_id', 'condition_start_datetime']]


############ drug
############ drug
print('Drug processing...')
drug = mpd.read_csv(os.path.join(dpath, 'drug_exposure.csv'))
drug.columns = [i.lower() for i in drug.columns]
drug['drug_concept_id'] = drug['drug_concept_id'].astype(int)
drug = drug[(drug['drug_concept_id'] != 0) & (drug['drug_concept_id'].isin(co['concept_id']))]
drug['drug_exposure_start_datetime'] = drug['drug_exposure_start_datetime'].fillna(drug['drug_exposure_start_date'])
drug['drug_exposure_end_datetime'] = drug['drug_exposure_end_datetime'].fillna(drug['drug_exposure_end_date'])
drug = drug[['person_id', 'drug_concept_id', 'drug_exposure_start_datetime', 'drug_exposure_end_datetime']]
drug = drug[
    (drug['drug_exposure_start_datetime'] >= '1900-01-01') & 
    (drug['drug_exposure_start_datetime'] <= '2260-12-31') & 
    (drug['drug_exposure_end_datetime'] >= '1900-01-01') & 
    (drug['drug_exposure_end_datetime'] <= '2260-12-31') 
]
drug = mpd.merge(
    drug,
    person[['person_id', 'year_of_birth']],
    on='person_id', how='left'
)
drug = drug[
    drug['drug_exposure_start_datetime'].apply(lambda x: x[:4]).astype(int) - drug['year_of_birth'] >= 18]
drug = drug[['person_id', 'drug_concept_id', 'drug_exposure_start_datetime']]


############ meas
############ meas
print('Meas processing...')
meas = mpd.read_csv(os.path.join(dpath, 'measurement.csv'))
meas.columns = [i.lower() for i in meas.columns]
meas['measurement_concept_id'] = meas['measurement_concept_id'].astype(int)
meas = meas[(meas['measurement_concept_id'] != 0) & (meas['measurement_concept_id'].isin(co['concept_id']))]
meas = meas[
    (meas['value_source_value'].notnull()) & 
    (meas['measurement_datetime'] >= '1900-01-01') & 
    (meas['measurement_datetime'] <= '2260-12-31') ]
meas = mpd.merge(
    meas,
    person[['person_id', 'year_of_birth']],
    on='person_id', how='left'
)

# Get one record for every hour
meas = meas[
    meas['measurement_datetime'].apply(lambda x: x[:4]).astype(int) - meas['year_of_birth'] >= 18]
meas['measurement_datehour'] = mpd.to_datetime(meas['measurement_datetime']).dt.floor('H')
meas['row'] = meas.groupby(
    ['person_id', 'measurement_concept_id', 'measurement_datehour']
)['value_source_value'].cumcount()+1
meas = meas[meas['row'] == 1].drop('row', axis=1)
meas = meas[[
    'person_id', 'measurement_concept_id', 'measurement_datetime', 
    'value_source_value', 'unit_source_value']]

# Measurements with numerical values --> Assign decile index (>=10 records)
meas['value_source_value'] = mpd.to_numeric(meas['value_source_value'], errors='coerce')
meas_with_numeric_value = meas[
    (meas['value_source_value'].notnull()) & (meas['unit_source_value'] != 'nan')]
meas_without_numeric_value = meas[
    (meas['value_source_value'].isnull()) | (meas['unit_source_value'] == 'nan')]

concept_unit_count = meas_with_numeric_value.groupby([
    'measurement_concept_id', 'unit_source_value'])['person_id'].count()
concept_unit_count_over10 = list(concept_unit_count[concept_unit_count >= 10].index)
meas_with_numeric_value = meas_with_numeric_value[meas_with_numeric_value.apply(
    lambda x: (x['measurement_concept_id'], x['unit_source_value']) in concept_unit_count_over10, 
    axis=1)]

# Quantile calculation
print('Quantile processing...')
meas_quantiles = meas_with_numeric_value.groupby(
    ['measurement_concept_id', 'unit_source_value'])['value_source_value'].apply(
    lambda x: x.quantile(np.linspace(0, 1, 11))
)
meas_quantiles = pd.DataFrame(np.concatenate(
    [np.array([i for i in meas_quantiles.index]), 
    np.array(meas_quantiles).reshape(-1, 1)], axis=1), 
    columns=['measurement_concept_id', 'unit_source_value', 'quantile', 'value_source_value'])
meas_quantiles = meas_quantiles.groupby(
    ['measurement_concept_id', 'unit_source_value'])['value_source_value'].apply(
        lambda x: np.array(x).astype(float))

# Save quantile information
mq_table = meas_quantiles.reset_index()
for i in range(11):
    mq_table[f'quantile_{i}'] = mq_table['value_source_value'].apply(lambda x: x[i])
mq_table.drop('value_source_value', axis=1).to_csv(os.path.join(spath, 'meas_quantiles.csv'), index=None)
print('Done')

# Assign quantile for each measurement
print('Assigning quantile...')
meas_quantiles = meas_quantiles.reset_index()
meas_quantiles['measurement_concept_id'] = meas_quantiles['measurement_concept_id'].astype(int)
meas_quantiles = meas_quantiles.set_index(['measurement_concept_id', 'unit_source_value'])['value_source_value']

def quantile_assign(concept_id, unit, value):
    use_quantile = meas_quantiles.loc[(concept_id, unit)]
    quantile = max(0, bisect.bisect_left(use_quantile, value) - 1)
    return quantile

meas_with_numeric_value['quantile'] = meas_with_numeric_value.apply(
    lambda x: quantile_assign(x['measurement_concept_id'], x['unit_source_value'], x['value_source_value']),
    axis=1
)
meas_with_numeric_value['measurement_concept_id'] = meas_with_numeric_value['measurement_concept_id'].astype(str) + \
    '_' + meas_with_numeric_value['quantile'].astype(str)
print('Done')

meas_with_numeric_value = meas_with_numeric_value[
    ['person_id', 'measurement_concept_id', 'measurement_datetime']]
meas_without_numeric_value = meas_without_numeric_value[
    ['person_id', 'measurement_concept_id', 'measurement_datetime']]
meas = mpd.concat([
    meas_with_numeric_value, meas_without_numeric_value
]).reset_index(drop=True)


############ proc
############ proc
print('Proc processing...')
proc = mpd.read_csv(os.path.join(dpath, 'procedure_occurrence.csv'))
proc.columns = [i.lower() for i in proc.columns]
proc['procedure_concept_id'] = proc['procedure_concept_id'].astype(int)
proc = proc[(proc['procedure_concept_id'] != 0) & (proc['procedure_concept_id'].isin(co['concept_id']))]
proc['procedure_datetime'] = proc['procedure_datetime'].fillna(proc['procedure_date'])
proc = mpd.merge(
    proc,
    person[['person_id', 'year_of_birth']],
    on='person_id', how='left'
)
proc = proc[
    proc['procedure_datetime'].apply(lambda x: x[:4]).astype(int) - proc['year_of_birth'] >= 18]
proc = proc[
    (proc['procedure_datetime'] >= '1900-01-01') & 
    (proc['procedure_datetime'] <= '2260-12-31') 
]
proc = proc[['person_id', 'procedure_concept_id', 'procedure_datetime']]
print('Done')



# Merge all records
print('Merging all records...')
files = {
    'cond': cond,
    'drug': drug,
    'meas': meas,
    'proc': proc,
}
for f, loadf in files.items():
    loadf['domain'] = f
    loadf.columns = ['record_datetime' if 'datetime' in c else c for c in loadf.columns]
    loadf.columns = ['concept_id' if 'concept_id' in c else c for c in loadf.columns]
    loadf = loadf[['person_id', 'domain', 'concept_id', 'record_datetime']]
    files[f] = loadf


allrecords = mpd.concat([v for v in files.values()]).sort_values(['person_id', 'record_datetime'])
regex_datetime = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$')
regex_date = re.compile(r'^\d{4}-\d{2}-\d{2}$')
allrecords = allrecords[allrecords['record_datetime'].apply(
    lambda x: bool(regex_datetime.match(x)) or bool(regex_date.match(x))
)]
allrecords['record_datetime'] = mpd.to_datetime(
    allrecords['record_datetime'].apply(
        lambda x: str(x) if ':' in x else str(x) + ' 00:00:00'))
allrecords = allrecords.drop_duplicates().reset_index(drop=True)
allrecords.to_csv(os.path.join(spath, f'allrecords.csv'), index=None)
print('Done')


# Train / Valid / Test split
# Remove patients with records less than 10
print('Splitting train / valid / test...')
code_count = allrecords.groupby('person_id')['domain'].count()
allrecords = allrecords[allrecords['person_id'].isin(
    list(code_count[code_count >= 10].index)
)]
allrecords_pids = allrecords['person_id'].unique()

# Train / Valid / Test split -> 70:15:15
train_id, test_id = train_test_split(allrecords_pids, test_size=0.3, random_state=50)
valid_id, test_id = train_test_split(test_id, test_size=0.5, random_state=50)
merge_id = {
    'train': train_id,
    'valid': valid_id,
    'test': test_id
}
for k, v in merge_id.items():
    _id = pd.DataFrame(columns=['person_id'])
    _id['person_id'] = list(v)
    _id.to_csv(os.path.join(spath, f'{k}_id.csv'), index=None)
print('Done')