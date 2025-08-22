"""
    Yield all using concept IDs and their neighboring concept IDs.
    Make chunk files of concept IDs to use call GPT API with multiple processes. 
    Input:
        (condition_occurrence.csv, drug_exposure.csv, measurement.csv, procedure_occurrence.csv)
        for all hospitals.
    Output:
        1. data/concepts/tmp/final_concepts.csv
        2. data/concepts/tmp/[Condition, Drug, Measurement, Procedure]_final_concepts.csv
        3. data/concepts/tmp/[Condition, Drug, Measurement, Procedure]/final_concepts_chunk_[0-19].csv
        
"""

import modin.pandas as mpd
import pandas as pd
import numpy as np
import ray
import os
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--num_chunks", type=int) 
parser.add_argument("--num_cpus", type=int) 
args = parser.parse_args()
ray.init(num_cpus=args.num_cpus)

abspath = str(Path(__file__).resolve().parent.parent.parent)
dpath = os.path.join(abspath, 'data')
cpath = os.path.join(abspath, 'data/concepts')
cpath_tmp = os.path.join(abspath, 'data/concepts/tmp')
spath = os.path.join(abspath, 'usedata')
os.makedirs(cpath_tmp, exist_ok=True)
os.makedirs(spath, exist_ok=True)


co = mpd.read_csv(os.path.join(cpath, 'CONCEPT.csv'), delimiter='\t')
co = co[co['domain_id'].isin(['Condition', 'Drug', 'Measurement', 'Procedure'])]
cr = mpd.read_csv(os.path.join(cpath, 'CONCEPT_RELATIONSHIP.csv'), delimiter='\t')
cr = cr[
    (cr['concept_id_1'].isin(co['concept_id'])) & 
    (cr['concept_id_2'].isin(co['concept_id']))
]

# The below file contains concept_ids in use.
# mimic.columns = ["concept_id"]
# If you have, add concept_ids from other institutions 
all_concepts = []
for hosp in ('mimic', 'ehrshot', 'snuh'):
    alltable = []
    for table in (
        'condition_occurrence', 'drug_exposure', 'measurement', 'procedure_occurrence'
    ):
        # Extract concept IDs from each table
        tmp = mpd.read_csv(os.path.join(dpath, f'data/{hosp}/{table}.csv'))
        tmp = tmp[f'{table.split("_")[0]}_concept_id'].value_counts().reset_index()
        tmp['domain'] = table
        tmp.columns = ['concept_id', 'count', 'domain']
        alltable.append(tmp)
    concept_hosp = mpd.concat(alltable)
    concept_hosp = mpd.merge(
        concept_hosp, co[['concept_id', 'concept_name']],
        on='concept_id', how='left'
    ).dropna()
    all_concepts.append(concept_hosp)
all_concepts = mpd.concat(all_concepts)['concept_id'].unique()

# Extract all neighbors of current concept IDs
cr1 = cr[cr['concept_id_1'].isin(all_concepts)]
use_concept_ids = cr1['concept_id_2'].unique()
co = co.set_index('concept_id').loc[use_concept_ids].reset_index()

categories = ['Condition', 'Drug', 'Measurement', 'Procedure']
co_category = {
    category: co[co['domain_id'] == category] for category in categories
}

# Add decile information for each measurement concept ID
meas_with_deciles = []
for n, i in enumerate(['1st', '2nd', '3rd'] + [f'{j}th' for j in range(4, 11)]):
    meas_tmp = co_category['Measurement'].copy()
    meas_tmp['concept_name'] = meas_tmp['concept_name'].apply(lambda x: f"{x} ({i} decile)")
    meas_tmp['concept_id'] = meas_tmp['concept_id'].apply(lambda x: f"{x}_{n}")
    meas_with_deciles.append(meas_tmp)
co_category['Measurement'] = mpd.concat([co_category['Measurement']] + meas_with_deciles)
for name, table in co_category.items():
    table[['concept_id', 'concept_name']].to_csv(
        os.path.join(cpath_tmp, f'{name}_final_concepts.csv'), index=None)

# Merge all concept IDs from all domains
allconcepts = mpd.concat([
    mpd.read_csv(os.path.join(cpath_tmp, f'{name}_final_concepts.csv')) \
        for name in categories]).reset_index(drop=True)
allconcepts.to_csv(os.path.join(cpath_tmp, "final_concepts.csv"), index=None)

# This process is for using GPT API (splitting)
for name in categories:
    os.makedirs(os.path.join(cpath_tmp, name), exist_ok=True)
    tmp = pd.read_csv(os.path.join(cpath_tmp, f'{name}_final_concepts.csv'))
    for n, i in enumerate(np.array_split(tmp, args.num_chunks)):
        i.to_csv(os.path.join(cpath_tmp, name, f'final_concepts_chunk_{n}.csv'), index=None)
    
    

