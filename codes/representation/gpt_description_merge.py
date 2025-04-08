"""
    Merge all descriptions.
    Input:
        1. usedata/descriptions/[Condition, Drug, Measurement, Procedure]/concept_description_[0-19].csv
    Output:
        1. usedata/descriptions/all_descriptions.csv
        
"""

import os
import modin.pandas as mpd
import ray
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--num_cpus", type=int, default=32, 
                    help='num cpus') 
args = parser.parse_args()

ray.init(num_cpus=args.num_cpus)

abspath = str(Path(__file__).resolve().parent.parent.parent)
dpath = os.path.join(abspath, 'usedata/descriptions')

alldes = []
for domain in ('Condition', 'Drug', 'Measurement', 'Procedure'):
    fpath = os.path.join(dpath, domain)
    des = mpd.concat([mpd.read_csv(os.path.join(fpath, f)) for f in os.listdir(fpath)])
    des['row'] = (des.groupby(['concept_id']).cumcount()+1)
    des = des[des['row'] == 1].drop(columns=['row'])
    alldes.append(des)
alldes = mpd.concat(alldes)
alldes['concept_id'] = alldes['concept_id'].astype(str)
alldes.to_csv(os.path.join(dpath, 'all_descriptions.csv'), index=None)

