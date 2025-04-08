"""
    Tokenize descriptions for all concept IDs.
    Input:
        1. usedata/descriptions/all_descriptions.csv
    Output:
        1. usedata/descriptions/tokens.pkl
        
"""

import os
import modin.pandas as mpd
import ray
from transformers import (
    DebertaTokenizer, )
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--num_cpus", type=int, default=32, 
                    help='num cpus') 
args = parser.parse_args()

import sys
from pathlib import Path
abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)
from src.utils import picklesave


def tokensave():
    ray.init(num_cpus=int(args.num_cpus / 8))

    abspath = str(Path(__file__).resolve().parent.parent.parent)
    cpath_tmp = os.path.join(abspath, 'usedata/descriptions')
    rpath = os.path.join(abspath, 'results/representation')
    lmpath = os.path.join(abspath, 'language_models')
    os.makedirs(rpath, exist_ok=True)

    # Load concept data
    allconcepts = mpd.read_csv(os.path.join(cpath_tmp, "all_descriptions.csv"))

    # Load DeBERTa model and tokenizer
    # We modified max_seq_length from 512 to 2048
    model_path = os.path.join(lmpath, 'deberta')
    if os.path.exists(model_path):
        tokenizer = DebertaTokenizer.from_pretrained(model_path)
    else:
        tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")

    # Tokenize and create dataset
    inputs = tokenizer(
        list(allconcepts['concept_description']), padding=True, return_tensors="pt")
    picklesave(inputs, os.path.join(cpath_tmp, 'tokens.pkl'))


if __name__ == "__main__":
    tokensave()
