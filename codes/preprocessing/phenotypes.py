import os
import ray
import modin.pandas as mpd
import argparse
import numpy as np
import sys
from pathlib import Path
abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)
from src.utils import picklesave

parser = argparse.ArgumentParser()
parser.add_argument('--num_cpus', type=int, default=64)
args = parser.parse_args()

ray.init(num_cpus=args.num_cpus)
abspath = str(Path(__file__).resolve().parent.parent.parent)
dpath = os.path.join(abspath, 'data')
upath = os.path.join(abspath, 'usedata')
co = mpd.read_csv(os.path.join(dpath, 'concepts/CONCEPT.csv'), delimiter='\t')
co9 = co[co['vocabulary_id'].str.contains('ICD9', na=False)]
co9['concept_code'] = co9['concept_code'].astype(str)
co9['concept_code'] = co9['concept_code'].apply(lambda x: x.replace('.', ''))
cr = mpd.read_csv(os.path.join(dpath, 'concepts/CONCEPT_RELATIONSHIP.csv'), delimiter='\t')
ca = mpd.read_csv(os.path.join(dpath, 'concepts/CONCEPT_ANCESTOR.csv'), delimiter='\t')
ph = mpd.read_csv(os.path.join(upath, 'icd9ccs.csv'))

ph_ids = {}
for i in ph.index:
    icds = ph.loc[i]['ICD9'].split(' ')
    cids = co9[co9['concept_code'].isin(icds)]['concept_id'].values
    cid_candidates = cr[cr['concept_id_1'].isin(cids)]
    cids = np.concatenate([
        cids,
        cid_candidates[cid_candidates['relationship_id'].isin(['Mapped from', 'Maps to'])]['concept_id_2'].values
    ])
    cids = np.concatenate([
        cids,
        ca[ca['ancestor_concept_id'].isin(cids)]['descendant_concept_id'].values # child ids
    ])
    ph_ids[ph.loc[i]['Phenotype']] = cids

picklesave(ph_ids, os.path.join(upath, 'phenotypes.pkl'))