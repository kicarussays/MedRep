import os
import ray
import json
import modin.pandas as mpd
import warnings
import argparse

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=Warning)


parser = argparse.ArgumentParser()
parser.add_argument('--num_cpus', type=int, default=64)
parser.add_argument('--hospital', type=str)
args = parser.parse_args()


# Set absolute path
from pathlib import Path
abspath = str(Path(__file__).resolve().parent.parent.parent)
dpath = os.path.join(abspath, f'data/{args.hospital}/')
spath = os.path.join(abspath, f'usedata/{args.hospital}/')
cpath = os.path.join(abspath, 'data/concepts')

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
co['concept_name'] = co['concept_name'].str.lower()

person = mpd.read_csv(os.path.join(dpath, 'person.csv'))
visit = mpd.read_csv(os.path.join(dpath, 'visit_occurrence.csv'))
visit.columns = [i.lower() for i in visit.columns]
death = mpd.read_csv(os.path.join(dpath, 'death.csv'))
death.columns = [i.lower() for i in death.columns]


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

# Get inpatients age between 18 and 90
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
visit_inp_death1['row'] = visit_inp_death1.groupby(['person_id']).cumcount()+1
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
readmission = readmission[(readmission['visit_start_date'] - readmission['visit_end_date']).dt.days <= 15]
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

final_label = mpd.concat([MT, LLOS, RA])
final_label.to_csv(os.path.join(spath, 'visit_label.csv'), index=None)