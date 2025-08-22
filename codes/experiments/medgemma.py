import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import ray
import modin.pandas as mpd
import pandas as pd
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--rep', type=str, 
                    help='rep type', default='medrep')
parser.add_argument('--outcome', type=str, 
                    help='outcome', default='MT')
parser.add_argument('--seed', type=str, 
                    help='seed', default=100)
parser.add_argument('--ex', type=str, 
                    help='mimic or ehrshot', default='mimic')
args = parser.parse_args()
r = args.rep
o = args.outcome
s = args.seed
ex = '' if args.ex == 'mimic' else '_ex'


import sys
from pathlib import Path
abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)
from src.utils import pickleload, picklesave

model_id = "google/medgemma-27b-text-it"

abspath = str(Path(__file__).resolve().parent.parent.parent)
seq = pickleload(os.path.join(abspath, f'results/atts/behrt+rep_type={r}_aug=0_{o}_mimic{ex}_seq_{s}.pkl'))[:, 2:]
fea = pickleload(os.path.join(abspath, f'results/atts/behrt+rep_type={r}_aug=0_{o}_mimic{ex}_last_cls_{s}.pkl'))[:, 2:]
ray.init(
    num_cpus=64,
    _system_config={
        "object_spilling_config": json.dumps(
            {"type": "filesystem", "params": {
                "directory_path": os.path.join(abspath, "trashbin/ray_spill")
            }}
        )
    },
    object_store_memory=int(200*1024*1024*1024)
)
desc = mpd.read_csv(os.path.join(abspath, 'usedata/descriptions/all_descriptions.csv'))
desc['concept_id'] = desc['concept_id'].astype(str)
desc = desc.loc[desc[['concept_id']].drop_duplicates(keep='first').index]
desc = desc.set_index(['concept_id'], drop=True)
fn = {
    'Fx': 'fracture',
    'Sepsis': 'sepsis',
    'PNA': 'pneumonia',
    'UTI': 'urinary tract infection',
    'MI': 'myocardial infarction',
} # feature name

hosp = 'mimic' if ex == '' else 'ehrshot'
rpath = os.path.join(abspath, f'results/{hosp}/medgemma/')
os.makedirs(rpath, exist_ok=True)
fpath = f'behrt+rep_type={r}_aug=0_{o}_mimic{ex}_last_cls_{s}.pkl'
print(f'behrt+rep_type={r}_aug=0_{o}_mimic{ex}_lst_cls_{s}.pkl')
if os.path.exists(os.path.join(rpath, fpath)): sys.exit(0)

model = AutoModelForCausalLM.from_pretrained(
    "models--google--medgemma-27b-text-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "models--google--medgemma-27b-text-it",
    local_files_only=True)
top_values, top_indices = torch.topk(fea, k=100, dim=1)
top_tokens = torch.gather(seq, dim=1, index=top_indices)

vocab = torch.load(os.path.join(abspath, f'usedata/{hosp}/vocab.pt'))
vocab = pd.DataFrame(list(vocab.keys()), columns=['concept_id'])
count = pd.Series(torch.cat([torch.unique(i) for i in top_tokens])).value_counts()
count = count.loc[[i for i in count.index if i > 6]][:100]
count = vocab.loc[count.index.values]
final_feats = str(desc.loc[count['concept_id'].values]['concept_name'].to_list())
MyPrompt = f"We trained a machine learning model to predict **{fn[o]}** based on patients' past medical records." + \
    f"From this model, we extracted 100 important features. Now, we want to assess how strongly each of these features is related to **{fn[o]}**." + \
    f'Please evaluate each feature and categorize its relevance to **{fn[o]}** as one of the following: "Low", "Moderate", or "High".' + \
    "Make sure to provide an assessment for all 100 features — do not omit any. Please provide your answer in CSV format with two columns: [feature, assessment]. The extracted features are as follows:" + \
    final_feats
print(MyPrompt)

messages = [
    {
        "role": "system",
        "content": "You are a helpful medical assistant."
    },
    {
        "role": "user",
        "content": MyPrompt
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=20000, do_sample=False)
    generation = generation[0][input_len:]

decoded = tokenizer.decode(generation, skip_special_tokens=True)

abspath = str(Path(__file__).resolve().parent.parent.parent)
picklesave(decoded, os.path.join(rpath, fpath))