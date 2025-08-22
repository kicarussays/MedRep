import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import argparse
parser = argparse.ArgumentParser()

abspath = str(Path(__file__).resolve().parent.parent.parent)
spath = os.path.join(abspath, 'usedata/{hosp}/')
tpath = os.path.join(abspath, 'usedata/{hosp}/tokens/Finetuning')
rpath = os.path.join(abspath, f'results/mimic/finetuning/')
dpath = os.path.join(abspath, f'usedata/descriptions/')
ppath = os.path.join(abspath, f'results/mimic/pretraining/saved/')
rep_path = os.path.join(abspath, f'usedata/representation')
lpath = os.path.join(abspath, f'results/logits/')
os.makedirs(lpath, exist_ok=True)


for rep_type in ('medrep', 'description', 'medrep_lwf_False', 'medtok'):
    for hosp in ('mimic', 'ehrshot'):
        vocab = torch.load(os.path.join(spath.format(hosp=hosp), 'vocab.pt'))
        special_tokens_rep_path = os.path.join(rep_path, f'special_tokens_representation.npy')
        if os.path.exists(special_tokens_rep_path):
            special_tokens_rep = np.load(special_tokens_rep_path)
        else:
            special_tokens_rep = torch.randn(7, 768)
            np.save(
                os.path.join(rep_path, f'special_tokens_representation.npy'), 
                special_tokens_rep.numpy())

        # ray.init(num_cpus=64)
        desc = pd.read_csv(os.path.join(rep_path, 'concept_idx.csv'))
        desc['concept_id'] = desc['concept_id'].astype(str)
        desc = desc.set_index('concept_id')
        desc_index = pd.DataFrame(np.arange(desc.shape[0]), index=desc.index, columns=['idx'])
        special_tokens = ['[PAD]', '[MASK]', '[UNK]', '[CLS]', '[SEP]', 'M', 'F']
        special_tokens_index = pd.DataFrame(
            np.arange(desc.shape[0], desc.shape[0]+7),
            index=special_tokens, columns=['idx']
        )
        desc_index = pd.concat([desc_index, special_tokens_index])
        representation = np.load(os.path.join(rep_path, f'concept_representation_{rep_type}.npy'))
        representation = np.concatenate([representation, special_tokens_rep])
        selected_representation = representation[desc_index.loc[list(vocab.keys())]['idx'].values]
        selected_rep_path = os.path.join(rep_path, f'concept_representation_{rep_type}_{hosp}.npy')
        np.save(selected_rep_path, selected_representation)
