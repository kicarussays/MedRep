import os
import ray
import json
import numpy as np
import pandas as pd
import modin.pandas as mpd
import warnings
import argparse

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=Warning)

parser = argparse.ArgumentParser()
parser.add_argument('--num_cpus', type=int, default=64)
parser.add_argument('--hospital', type=str, default='ehrshot')
args = parser.parse_args()

import sys
from pathlib import Path
abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)
from src.vars import disease_ids


# Set absolute path
from pathlib import Path
abspath = str(Path(__file__).resolve().parent.parent.parent)
dpath = os.path.join(abspath, f'data/{args.hospital}/')
cpath = os.path.join(abspath, 'data/concepts')
spath = os.path.join(abspath, f'usedata/{args.hospital}/')
upath = os.path.join(abspath, f'usedata')

os.makedirs(dpath, exist_ok=True)
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
co = mpd.read_csv(os.path.join(cpath, 'CONCEPT.csv'), delimiter='\t')
co = co[(co['concept_name'].notnull())]
co['concept_name'] = co['concept_name'].str.lower()
co['concept_id'] = co['concept_id'].astype(str)

allrecords = mpd.read_csv(os.path.join(spath, 'allrecords_divided.csv'))
test_id = pd.read_csv(os.path.join(spath, 'test_id.csv'))
person = mpd.read_csv(os.path.join(dpath, 'person.csv'))
visit = mpd.read_csv(os.path.join(dpath, 'visit_occurrence.csv'))
visit.columns = [i.lower() for i in visit.columns]
death = mpd.read_csv(os.path.join(dpath, 'death.csv'))
death.columns = [i.lower() for i in death.columns]

if args.hospital == 'mimic':
    allrecords = allrecords[allrecords['person_id'].isin(test_id['person_id'])]
    visit = visit[visit['person_id'].isin(test_id['person_id'])]
allrecords['record_datetime'] = mpd.to_datetime(allrecords['record_datetime'])
visit['visit_start_date'] = mpd.to_datetime(visit['visit_start_date'])
visit['visit_end_date'] = mpd.to_datetime(visit['visit_end_date'])


# Anchor age applying (only for mimic data)
# year_of_birth must be set according to the column ("anchor_age") of original table "patients.csv"
if args.hospital == 'mimic':
    patients = mpd.read_csv(os.path.join(dpath, 'patients.csv'))
    person['subject_id'] = person['trace_id'].apply(lambda x: x.split(':')[1][:-1]).astype(int)
    person = mpd.merge(
        person, patients[['subject_id', 'anchor_age']],
        on='subject_id', how='left'
    )
    person['year_of_birth'] = person['year_of_birth'] - person['anchor_age']

# Datetime conversion
for table in (visit, death):
# for table in (visit, death, meas):
    for datetimecol in [i for i in table.columns if 'date' in i]:
        table[datetimecol] = mpd.to_datetime(table[datetimecol])

# Get inpatients age between 18 and 100
visit_inp = mpd.merge(
    visit[visit['visit_concept_id'].isin([9201, 262])],
    person[['person_id', 'gender_concept_id', 'year_of_birth', 'race_concept_id']], on='person_id', how='left'
)
visit_inp['age'] = mpd.to_datetime(visit_inp['visit_start_date']).dt.year - visit_inp['year_of_birth']
visit_inp = visit_inp[visit_inp['age'].between(18, 100, inclusive='both')]

# Merge death information
visit_inp_death = mpd.merge(
    visit_inp,
    death[['person_id', 'death_date', 'death_datetime']],
    on='person_id', how='left'
)
admission_cnt1 = len(visit_inp_death)
print(f'Adults admissions: {admission_cnt1}')

# Exclude admission if patient death or discharge occurred on the date of admission
visit_inp_death = visit_inp_death[~(visit_inp_death['visit_start_date'] >= visit_inp_death['death_date'])]
visit_inp_death = visit_inp_death[visit_inp_death['visit_start_date'] != visit_inp_death['visit_end_date']]
admission_cnt2 = len(visit_inp_death)
visit_inp_death_all = visit_inp_death.copy()
print(f'Adults admissions without death or discharge: {admission_cnt2} (-{admission_cnt1 - admission_cnt2})')



### Labling mortality and LLOS
# Exclude non-selected admissions among patients with multiple admissions
visit_inp_death1 = visit_inp_death.sort_values(['person_id', 'visit_start_date'], ascending=[True, False])
visit_inp_death1['row'] = visit_inp_death1.groupby('person_id').cumcount()+1
visit_inp_death1 = visit_inp_death1[visit_inp_death1['row'] == 1].reset_index(drop=True).drop(columns=['row'])


# In-hospital mortality
visit_inp_death1['mortality'] = (visit_inp_death1['death_date'].notnull()) & \
    (visit_inp_death1['death_date'].between(
        visit_inp_death1['visit_start_date'], 
        visit_inp_death1['visit_end_date'],
        inclusive='right'))

# Long LOS
visit_inp_death1['LLOS'] = (visit_inp_death1['visit_end_date'] - visit_inp_death1['visit_start_date']).dt.days >= 7


### Labling 30-day Readmission
# Exclude non-selected admissions among patients with multiple admissions
visit_inp_death2 = visit_inp_death.sort_values(['person_id', 'visit_start_date'], ascending=[True, False])
visit_inp_death2['row'] = visit_inp_death2.groupby(['person_id']).cumcount()+1
visit_inp_death2 = visit_inp_death2[visit_inp_death2['row'] != 1].reset_index(drop=True).drop(columns=['row'])
visit_inp_death2 = visit_inp_death.groupby(['person_id']).sample(1).reset_index(drop=True)

readmission = mpd.merge(
    visit_inp_death2[['person_id', 'visit_end_date', 'visit_end_datetime']], # Current admission
    visit_inp_death_all[['person_id', 'visit_start_date', 'visit_start_datetime']], # Next admission
    on='person_id', how='left'
)
readmission = readmission[readmission['visit_start_date'] > readmission['visit_end_date']].sort_values(['person_id', 'visit_start_datetime'])
readmission['admission_rank'] = readmission.groupby('person_id').cumcount()+1
readmission = readmission[readmission['admission_rank'] == 1].drop(['admission_rank'], axis=1)
readmission = readmission[(readmission['visit_start_date'] - readmission['visit_end_date']).dt.days <= 30]
visit_inp_death2['readmission'] = visit_inp_death2['person_id'].isin(readmission['person_id'])


### Merging
MT = visit_inp_death1[['person_id', 'visit_start_date', 'mortality']]
MT['visit_start_date'] = MT['visit_start_date'] + mpd.DateOffset(days=1)
MT['label_type'] = 'MT'
MT.columns = ['person_id', 'prediction_timepoint', 'label', 'label_type']

LLOS = visit_inp_death1[['person_id', 'visit_start_date', 'LLOS']]
LLOS['visit_start_date'] = LLOS['visit_start_date'] + mpd.DateOffset(days=1)
LLOS['label_type'] = 'LLOS'
LLOS.columns = ['person_id', 'prediction_timepoint', 'label', 'label_type']

RA = visit_inp_death2[['person_id', 'visit_end_date', 'readmission']]
RA['visit_end_date'] = RA['visit_end_date'] + mpd.DateOffset(days=1)
RA['label_type'] = 'RA'
RA.columns = ['person_id', 'prediction_timepoint', 'label', 'label_type']


### Disease labeling 
disease_ids['Fx'] = co[
        (co['concept_name'].str.contains('closed.*fracture')) & 
        (co['domain_id'] == 'Condition') & 
        (co['vocabulary_id'] == 'SNOMED')
    ]['concept_id'].values.tolist()

final_cohorts = []
for dx, ids in disease_ids.items():
    allrecords_case = allrecords[allrecords['concept_id'].isin(np.array(ids).astype(str))]

    visit_dx = mpd.merge(
        visit,
        allrecords_case[['person_id', 'record_datetime']],
        on='person_id', how='inner'
    )

    visit_dx = visit_dx[
        (visit_dx['record_datetime'].between(
            visit_dx['visit_start_date'], visit_dx['visit_end_date'] + mpd.DateOffset(days=1))) & 
        ((visit_dx['visit_end_date'] - visit_dx['visit_start_date']).dt.days >= 2)
    ][['person_id', 'record_datetime']].sort_values(
        ['person_id', 'record_datetime'], ascending=[True, False]).reset_index(drop=True)

    visit_dx = visit_dx.loc[visit_dx[['person_id']].drop_duplicates(keep='first').index]

    visit_over3 = visit[(visit['visit_end_date'] - visit['visit_start_date']).dt.days >= 2]
    allrecords_control = allrecords[
        (~allrecords['person_id'].isin(
            allrecords[allrecords['concept_id'].isin(np.array(ids).astype(str))]['person_id']
        )) & 
        (allrecords['person_id'].isin(visit_over3['person_id']))
    ].groupby('person_id').sample(1)

    visit_dx = visit_dx[['person_id', 'record_datetime']].reset_index(drop=True)
    visit_dx['record_datetime'] = visit_dx['record_datetime'].dt.date - mpd.DateOffset(days=1)
    visit_dx['label'] = True
    visit_dx['label_type'] = dx
    visit_dx.columns = ['person_id', 'prediction_timepoint', 'label', 'label_type']

    allrecords_control = allrecords_control[['person_id', 'record_datetime']].reset_index(drop=True)
    allrecords_control['record_datetime'] = allrecords_control['record_datetime'].dt.date
    allrecords_control['label'] = False
    allrecords_control['label_type'] = dx
    allrecords_control.columns = ['person_id', 'prediction_timepoint', 'label', 'label_type']

    final_cohort = mpd.concat([visit_dx, allrecords_control])
    if final_cohort.shape[0] == len(set(final_cohort['person_id'])):
        print(f"({dx}) Case-control Counts:")
        print(final_cohort['label'].value_counts())

        final_cohorts.append(final_cohort)
    else:
        print('Error')

final_label = mpd.concat([MT, LLOS, RA])
final_label.to_csv(os.path.join(spath, 'visit_label.csv'), index=None)


import sys
from tqdm import tqdm
abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)
from src.utils import pickleload
phe = pickleload(os.path.join(upath, 'phenotypes.pkl'))
all_phenotype_codes = np.concatenate([v for v in phe.values()]).astype(str)
all_phenotype_codes = set([str(i) for i in all_phenotype_codes])
for k, v in phe.items():
    phe[k] = set([str(i) for i in v])

pids = allrecords[allrecords['record_rank'] == 3]['person_id'].unique()
target_pids = []
prediction_timepoints = []
all_labels = []
for pid in tqdm(pids):
    precords = allrecords[allrecords['person_id'] == pid]
    target_visit = precords[precords['record_rank'] == 3]['visit_rank'].max()
    load_concept = set(precords[(precords['visit_rank'] == target_visit) & 
                                (precords['record_rank'] <= 2)]['concept_id'].values)
    if len(all_phenotype_codes & set(load_concept)) != 0: 
        continue
    load_concept = precords[(precords['visit_rank'] == target_visit) & 
                            (precords['record_rank'] > 2)]
    predict_timepoint = load_concept['record_datetime'].dt.date.iloc[0]
    load_concept = set(load_concept['concept_id'].values)
    label = np.array([len(v & load_concept) != 0 for k, v in phe.items()]).astype(int)
    
    target_pids.append(pid)
    prediction_timepoints.append(predict_timepoint)
    all_labels.append(label)

pheno_labels = pd.DataFrame()
pheno_labels['person_id'] = target_pids
pheno_labels['prediction_timepoint'] = prediction_timepoints

all_labels = np.stack(all_labels)
for n, k in enumerate(phe.keys()):
    pheno_labels[k] = all_labels[:, n]
pheno_labels.to_csv(os.path.join(spath, 'phenotypes.csv'), index=None)