import os
import sys
from pathlib import Path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', '-d', type=str, 
                    help='cpu or GPU Device Number', default=0)
parser.add_argument('--seed', type=int, 
                    help='Random seed', default=100)
parser.add_argument('--hospital', type=str, 
                    help='mimic or ehrshot', default='mimic')
parser.add_argument('--model', type=str, 
                    help='behrt-de, medbert', default='behrt')
parser.add_argument('--outcome', type=str, 
                    help='Outcomes', default='Sepsis')
parser.add_argument('--rep-type', type=str,
                    help='Select representation type', default='none')
parser.add_argument('--aug-times', type=int, 
                    help='How many times to augment', default=0)
parser.add_argument('--aug-ratio', type=float, 
                    help='How many times to augment', default=0.8)
parser.add_argument('--use-adapter', action='store_true',
                    help='use adapter')
parser.add_argument('--extract-attention-score', action='store_true',
                    help='use adapter')
args = parser.parse_args()

abspath = str(Path(__file__).resolve().parent.parent.parent)
spath = os.path.join(abspath, 'usedata/{hosp}/')
tpath = os.path.join(abspath, 'usedata/{hosp}/tokens/Finetuning')
rpath = os.path.join(abspath, f'results/{args.hospital}/finetuning/')
dpath = os.path.join(abspath, f'usedata/descriptions/')
ppath = os.path.join(abspath, f'results/mimic/pretraining/saved/')
rep_path = os.path.join(abspath, f'usedata/representation')
lpath = os.path.join(abspath, f'results/logits/')
os.makedirs(lpath, exist_ok=True)
os.makedirs(lpath.replace('logits', 'atts'), exist_ok=True)

param_option = f'rep_type={args.rep_type}_aug={args.aug_times}'
logit_path = os.path.join(
    lpath, 
    f'{args.model}+{param_option}_{args.outcome}_{args.hospital}')


import numpy as np
import pandas as pd
import torch
import yaml
import copy
from transformers import BertConfig
import ray
import modin.pandas as mpd
import torch.multiprocessing as mp
torch.set_num_threads(8)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)
from src.utils import seedset, pickleload, DotDict
from src.dataset import BinaryOutcomeDataset
from src.trainer import EHRClassifier
from src.baseline_models import EHRBertForSequenceClassification
configpath = os.path.join(abspath, f'configs/{args.model}.yaml')


def main():
    with open(configpath, 'r') as f: 
        config = yaml.safe_load(f)
        args.bs = config['trainconfig']['bs']
        args.lr = config['trainconfig']['lr']
        args.max_epoch = config['trainconfig']['max_epoch']
        config['bertconfig']['rep_type'] = args.rep_type
        config = DotDict(**config)
        
    device = f'cuda:{args.device}'
    seedset(args.seed)
    os.makedirs(os.path.join(rpath, f'logs'), exist_ok=True)
    os.makedirs(os.path.join(rpath, f'saved'), exist_ok=True)

    pretrained_param = f'all+rep_type={args.rep_type}'
    # pretrained_param = f'{args.group}+rep_type={args.rep_type}'
    args.logpth = os.path.join(rpath, f'logs/{args.model}_{param_option}_{args.outcome}_{args.hospital}.log')
    args.savepth = os.path.join(rpath, f'saved/{args.model}_{param_option}_{args.outcome}_{args.hospital}')

    vocab = torch.load(os.path.join(spath.format(hosp='mimic'), 'vocab.pt'))
    ex_vocab = torch.load(os.path.join(spath.format(hosp='ehrshot'), 'vocab.pt'))
    bertconfig = BertConfig(
        vocab_size=len(vocab), 
        problem_type='multi_label_classification' if args.outcome in ('Pheno', 'Drugrec') else 'single_label_classification',
        use_adapter=args.use_adapter,
        **config.bertconfig)

    # Loading representations for nn.Embedding layers
    neighbors_mapped = None
    selected_representation = None
    ex_selected_representation = None
    if args.rep_type != 'none':
        aug_option = '_aug' if args.aug_times != 0 else ''
        selected_rep_path = os.path.join(rep_path, f'concept_representation_{args.rep_type}_mimic{aug_option}.npy')
        ex_selected_rep_path = os.path.join(rep_path, f'concept_representation_{args.rep_type}_ehrshot.npy')
        selected_representation = np.load(selected_rep_path)
        ex_selected_representation = np.load(ex_selected_rep_path)

        if aug_option == '_aug':
            # Get neighbors indexes
            neighbors_mapped_path = os.path.join(rep_path, f'neighbors_mapped_vocab_{args.hospital}_{args.rep_type}.npy')
            neighbors_mapped = np.load(neighbors_mapped_path)
            

    # Loading training and test data
    all_ds = []
    for hosp in ('mimic', 'ehrshot'):
        if args.rep_type == 'none':
            token, label = pickleload(os.path.join(tpath.format(hosp=hosp), f'{args.outcome}_mimic.pkl'))
        else:
            token, label = pickleload(os.path.join(tpath.format(hosp=hosp), f'{args.outcome}_{hosp}.pkl'))

        token['age'] = torch.clamp(token['age'], min=0, max=config.bertconfig.max_age-1)
        if args.outcome in ('Pheno', 'Drugrec'):
            label = torch.Tensor(label).type(torch.FloatTensor)
            bertconfig.num_labels = len(label[0])
        else:
            label = torch.Tensor(label).type(torch.LongTensor)
        if args.model != 'behrt-de':
            token = {k: v for k, v in token.items() if k != 'domain'}
    
        dataset = BinaryOutcomeDataset(
            token, 
            outcomes=label,
            vocabulary=vocab, 
            ignore_special_tokens=True,
            neighbors_mapped=neighbors_mapped if hosp == 'mimic' else None,
            aug_times=args.aug_times,
            aug_ratio=args.aug_ratio,
        )
        all_ds.append(dataset)
    
    # Loading model
    model = EHRBertForSequenceClassification(bertconfig)
    load_pretrained = torch.load(os.path.join(ppath, f'{args.model}_{pretrained_param}.tar'),
                                map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in load_pretrained['model'].items():
        name = k.replace("module.", "")
        if 'concept_embeddings' not in name:
            new_state_dict[name] = v
    _load = model.load_state_dict(new_state_dict, strict=False)
    print(_load)

    # Loading representations to embedding layers
    if args.rep_type != 'none':
        model.bert.embeddings.concept_embeddings.weight.data = torch.Tensor(selected_representation)
        for param in model.bert.embeddings.parameters(): 
            param.requires_grad_(False)
        
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}")

    model = model.to(device)
    # Training classifier 
    Trainer = EHRClassifier(
        dataset=all_ds[0],
        ex_dataset=all_ds[1],
        logit_path=logit_path,
        vocab=vocab,
        ex_vocab=ex_vocab,
        selected_representation=None if args.rep_type == 'none' else selected_representation,
        ex_selected_representation=ex_selected_representation,
        args=args
    )

    if not args.extract_attention_score:
        Trainer.process(model, args)
    else:
        Trainer.extract_attention_score(model, args)
    

if __name__ == '__main__':
    main()


