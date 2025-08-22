import os
import sys
from pathlib import Path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, 
                    help='Random seed', default=100)
parser.add_argument('--model', type=str, 
                    help='behrt, behrt-de, medbert')
parser.add_argument('--rep-type', type=str, default='none',
                    help='Select representation type')
parser.add_argument('--group', type=str, default='all',
                    help='Select subgroup')

# argument for multi-gpu
parser.add_argument('--num-workers', type=int, default=4, help='')
parser.add_argument("--gpu-devices", type=int, nargs='+', default=None, help="")
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
# parser.add_argument('--dist-url', default='tcp://127.0.0.1:4567', type=str, help='')
parser.add_argument('--dist-port', default=4567, type=int, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world-size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
args = parser.parse_args()


def finish_check(path):
    f = open(path, 'r')
    lines = f.readlines()
    flag = False
    for line in lines:
        line = line.strip()  
        if 'Early stopping activated' in line or 'Epoch [50/50]' in line:
            flag = True
            break
    return flag


abspath = str(Path(__file__).resolve().parent.parent.parent)
spath = os.path.join(abspath, f'usedata/mimic/')
tpath = os.path.join(abspath, f'usedata/mimic/tokens/')
dpath = os.path.join(abspath, f'usedata/descriptions/')
rpath = os.path.join(abspath, f'results/mimic/pretraining/')
rep_path = os.path.join(abspath, f'usedata/representation')

param_option = f'rep_type={args.rep_type}'
args.logpth = os.path.join(rpath, f'logs/{args.model}_{args.group}+{param_option}.log')
args.savepth = os.path.join(rpath, f'saved/{args.model}_{args.group}+{param_option}.tar')
if os.path.exists(args.logpth):
    if finish_check(args.logpth):
        print("Fin.")
        sys.exit(1)

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import yaml
from transformers import BertConfig
import ray
import modin.pandas as mpd
from sklearn.model_selection import train_test_split

torch.set_num_threads(32)


gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
os.environ["NCCL_TIMEOUT"] = '1200'
os.environ["MASTER_PORT"] = '12355'
ngpus_per_node = torch.cuda.device_count()
args.world_size = ngpus_per_node * args.world_size
args.dist_url = f'tcp://127.0.0.1:{args.dist_port}'


abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)
from src.utils import seedset, pickleload, DotDict, finish_check
from src.dataset import MLM_Dataset
from src.trainer import EHRTrainer
from src.baseline_models import (
    BehrtForMaskedLM,
    MedbertForMaskedLM)
configpath = os.path.join(abspath, f'configs/{args.model}.yaml')


def main():
    with open(configpath, 'r') as f: 
        config = yaml.safe_load(f)
        args.bs = config['trainconfig']['bs']
        args.lr = config['trainconfig']['lr']
        args.max_epoch = config['trainconfig']['max_epoch']
        config['bertconfig']['rep_type'] = args.rep_type
        config = DotDict(**config)
            
    print('\n', args, '\n')
    seedset(args.seed)
    os.makedirs(os.path.join(rpath, f'logs'), exist_ok=True)
    os.makedirs(os.path.join(rpath, f'saved'), exist_ok=True)
    
    vocab = torch.load(os.path.join(spath, 'vocab.pt'))
    load_dataloader = {}
    allgroup = [f'group{i}' for i in range(1, 9)]
    if args.group == "all":
        allgroup_data = [pickleload(os.path.join(tpath, f'Pretraining_tokens_{g}.pkl')) for g in allgroup]
        token = {}
        for k in allgroup_data[0].keys():
            token[k] = torch.cat([d[k] for d in allgroup_data])
    else:
        token = pickleload(os.path.join(tpath, f'Pretraining_tokens_{args.group}.pkl'))
    token['age'] = torch.clamp(token['age'], min=0, max=config.bertconfig.max_age-1)
    if args.model != 'behrt-de':
        token = {k: v for k, v in token.items() if k != 'domain'}
    if args.model == 'medbert':
        if args.group == "all":
            allgroup_data = [torch.Tensor(pickleload(os.path.join(tpath, f'Pretraining_plos_{g}.pkl'))) for g in allgroup]
            token['plos_target'] = torch.cat(allgroup_data)
        else:
            token['plos_target'] = torch.Tensor(
                pickleload(os.path.join(tpath, f'Pretraining_plos_{args.group}.pkl')))

    train_id, test_id = train_test_split(
        np.arange(token['concept'].shape[0]), test_size=0.2, random_state=args.seed)
    
    train_test_ds = []
    for ids in (train_id, test_id):
        _token = {k: v[ids] for k, v in token.items()}
        dataset = MLM_Dataset(
            _token, 
            vocabulary=vocab, 
            masked_ratio=config.datasetconfig.masked_ratio,
            ignore_special_tokens=True,)
        train_test_ds.append(dataset)
    load_dataloader['train'] = train_test_ds[0]
    load_dataloader['valid'] = train_test_ds[1]


    bertconfig = BertConfig(vocab_size=len(vocab), **config.bertconfig)
    if args.model in ('behrt', 'behrt-de'):
        model = BehrtForMaskedLM(bertconfig)
    elif args.model == 'medbert':
        model = MedbertForMaskedLM(bertconfig)
    else:
        assert "Cannot applicable the model"
    
    if os.path.exists(args.savepth):
        print('\n\nSaved file exists.\n\n')
        checkpoint = torch.load(args.savepth, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        args.first_iter = checkpoint["epoch"] + 1
        if args.first_iter >= 100: 
            print('Fin.')
            sys.exit(1)
        model.load_state_dict(new_state_dict)
    else:
        args.first_iter = 0
    
    if args.rep_type != 'none':
        selected_representation_path = os.path.join(rep_path, f'concept_representation_{args.rep_type}_mimic.npy')
        representation = np.load(selected_representation_path)
                
        model.bert.embeddings.concept_embeddings = torch.nn.Embedding(*representation.shape)
        model.bert.embeddings.concept_embeddings.weight.data = torch.Tensor(representation)
        model.bert.embeddings.concept_embeddings.weight.requires_grad_(False)

    Trainer = EHRTrainer(
        model=model,
        train_dataset=load_dataloader['train'],
        valid_dataset=load_dataloader['valid'],
        args=args,
    )

    mp.spawn(Trainer.train, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


if __name__ == '__main__':
    main()

