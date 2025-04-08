import os
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.distance import cdist
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_cpus", type=int, default=32, 
                    help='num cpus') 
args = parser.parse_args()

import ray
import modin.pandas as mpd
from pathlib import Path
import sys
abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)

abspath = str(Path(__file__).resolve().parent.parent.parent)
upath = os.path.join(abspath, 'usedata')
dpath = os.path.join(abspath, 'usedata/descriptions')
rpath = os.path.join(abspath, 'usedata/representation')
cpath = os.path.join(abspath, 'data/concepts')
ray.init(num_cpus=args.num_cpus)
alldesc = mpd.read_csv(os.path.join(dpath, 'all_descriptions.csv'))
alldesc['concept_id'] = alldesc['concept_id'].astype(str)
co = mpd.read_csv(os.path.join(cpath, 'CONCEPT.csv'), delimiter='\t')
co['concept_id'] = co['concept_id'].astype(str)
reps = {
    rep: np.load(os.path.join(rpath, f'concept_representation_{rep}.npy'))
    # for rep in ('biobert', 'description', )
    for rep in ('biobert+gnn', 'description+gnn')
}
alldesc = mpd.merge(
    alldesc, co[['concept_id', 'domain_id']], on='concept_id', how='left'
)

vocabs = {
    k: torch.load(os.path.join(upath, f'{k}/vocab.pt')) for k in ['mimic', 'ehrshot']
}

if os.path.exists(os.path.join(dpath, 'included_descriptions.csv')):
    included_desc = mpd.read_csv(os.path.join(dpath, 'included_descriptions.csv'))
    included_desc = included_desc.set_index('index')
else:
    included_concepts = list(
        set(list(vocabs['mimic'].keys()) + \
            list(vocabs['ehrshot'].keys())))
    included_desc = alldesc[alldesc['concept_id'].isin(included_concepts)]

    for k in ['mimic', 'ehrshot']:
        included_desc[k] = included_desc['concept_id'].isin(list(vocabs[k].keys()))
    included_desc['idx'] = np.arange(included_desc.shape[0])
    included_desc.reset_index().to_csv(os.path.join(dpath, 'included_descriptions.csv'), index=None)


@ray.remote
def ray_cdist(whole, chunk):
    dist = cdist(chunk, whole, metric='euclidean')
    return dist


@ray.remote
def ray_neighbors(chunk):
    all_neighbors = []
    for i in tqdm(chunk):
        _included_desc = included_desc.copy()
        _included_desc['dist'] = i
        _included_desc = _included_desc.sort_values(['dist'])
        neighbors = np.concatenate([
            # Extract 20 neighbors for each hospital
            _included_desc[_included_desc[hosp]].iloc[1:11]['idx'].values \
                for hosp in (['mimic', 'ehrshot'])])
        all_neighbors.append(neighbors)
    return all_neighbors


for k, v in reps.items():
    chunks = np.array_split(v[included_desc.index], 64)
    all_dist = ray.get([ray_cdist.remote(v[included_desc.index], chunk) for chunk in tqdm(chunks)])
    all_dist = np.concatenate(all_dist)

    all_neighbors = ray.get([ray_neighbors.remote(chunk) for chunk in np.array_split(all_dist, 64)])
    all_neighbors = np.concatenate(all_neighbors)
    np.save(
        os.path.join(rpath, f'neighbors_{k}.npy'),
        all_neighbors
    )

