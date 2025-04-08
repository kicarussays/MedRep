"""
    Yield BioBERT representations of all concept IDs.
    Input:
        1. usedata/descriptions/all_descriptions.csv
    Output:
        1. usedata/representation/concept_representation_biobert.npy
        
"""

import os
import numpy as np
import modin.pandas as mpd
import ray
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path
import argparse
import torch.distributed as dist


parser = argparse.ArgumentParser()
parser.add_argument("--num_cpus", type=int, default=32, 
                    help='num cpus') 
parser.add_argument("-d", "--device", type=int, default=1, 
                    help='representation type') 
parser.add_argument("--bs", type=int, default=2048, 
                    help='Batch size') 
args = parser.parse_args()

ray.init(num_cpus=args.num_cpus)
device = f'cuda:{args.device}'

abspath = str(Path(__file__).resolve().parent.parent.parent)
cpath_tmp = os.path.join(abspath, 'usedata/descriptions')
spath = os.path.join(abspath, 'usedata')
lmpath = os.path.join(abspath, 'language_models')
os.makedirs(spath, exist_ok=True)

# Load concept data
allconcepts = mpd.read_csv(os.path.join(cpath_tmp, "all_descriptions.csv"))

# Load BioBERT model and tokenizer
model_path = os.path.join(lmpath, 'biobert')
if os.path.exists(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
else:
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

# Move model to GPU if available
model.to(device)

# Tokenize and create dataset
inputs = tokenizer(
    list(allconcepts['concept_name']), padding=True, truncation=True, return_tensors="pt")
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])

# Define DataLoader for minibatch processing
batch_size = args.bs
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Process minibatches
cls_representations_list = []
with torch.no_grad():
    for batch in tqdm(dataloader):
        input_ids, attention_mask = [tensor.to(device) for tensor in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        cls_representations = outputs.last_hidden_state[:, 0, :].cpu().detach()
        cls_representations_list.append(cls_representations)

# Concatenate all minibatch CLS representations
cls_representations = torch.cat(cls_representations_list, dim=0)

# Print the CLS representations
os.makedirs(os.path.join(spath, 'representation'), exist_ok=True)
np.save(os.path.join(spath, 'representation', 'concept_representation_biobert.npy'), 
        cls_representations.cpu().detach().numpy())


