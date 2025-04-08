import os
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import argparse
import yaml
from transformers import BertConfig
import ray
import modin.pandas as mpd

torch.set_num_threads(32)

parser = argparse.ArgumentParser()
parser.add_argument('--device', '-d', type=str, 
                    help='cpu or GPU Device Number', default=0)
parser.add_argument('--seed', type=int, 
                    help='Random seed', default=100)
parser.add_argument('--hospital', type=str, 
                    help='hospital', default='mimic')
parser.add_argument('--model', type=str, 
                    help='behrt, behrt-de, medbert, ethos', default='behrt-de')
parser.add_argument('--rep-type', type=str, default='none',
                    help='Select representation type')
parser.add_argument('--smooth', action='store_true',
                    help='loss smoothing option')
parser.add_argument('--multi-gpu', action='store_true', 
                    help='Use multi gpu?')

# argument for multi-gpu
parser.add_argument('--num-workers', type=int, default=4, help='')
parser.add_argument("--gpu-devices", type=int, nargs='+', default=None, help="")
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world-size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
args = parser.parse_args()

if args.multi_gpu:
    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    os.environ["NCCL_TIMEOUT"] = '1200'
    os.environ["MASTER_PORT"] = '12355'
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size


import sys
from pathlib import Path
abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)
from src.utils import seedset, pickleload, DotDict
from src.dataset import MLM_Dataset, EthosDataset
from src.trainer import EHRTrainer, EthosTrainer
from src.baseline_models import (
    BehrtForMaskedLM,
    MedbertForMaskedLM, Ethos)
configpath = os.path.join(abspath, f'configs/{args.model}.yaml')


abspath = str(Path(__file__).resolve().parent.parent.parent)
spath = os.path.join(abspath, f'usedata/{args.hospital}/')
tpath = os.path.join(abspath, f'usedata/{args.hospital}/tokens/')
dpath = os.path.join(abspath, f'usedata/descriptions/')
rpath = os.path.join(abspath, f'results/{args.hospital}/pretraining/')
rep_path = os.path.join(abspath, f'usedata/representation')


def main():
    with open(configpath, 'r') as f: 
        config = yaml.safe_load(f)
        args.bs = config['trainconfig']['bs']
        args.lr = config['trainconfig']['lr']
        args.max_epoch = config['trainconfig']['max_epoch']
        config['bertconfig']['smooth'] = args.smooth
        config['bertconfig']['rep_type'] = args.rep_type
        config = DotDict(**config)
            
    seedset(args.seed)
    os.makedirs(os.path.join(rpath, f'logs'), exist_ok=True)
    os.makedirs(os.path.join(rpath, f'saved'), exist_ok=True)

    param_option = f'rep_type={args.rep_type}_smooth={args.smooth}'
    args.logpth = os.path.join(rpath, f'logs/{args.model}+{param_option}.log')
    args.savepth = os.path.join(rpath, f'saved/{args.model}+{param_option}.tar')
    
    vocab = torch.load(os.path.join(spath, 'vocab.pt'))
    load_dataloader = {}
    for file in ('train', 'valid'):
        if args.model != 'ethos':
            token = pickleload(os.path.join(tpath, f'Pretraining_tokens_{file}.pkl'))
            token['age'] = torch.clamp(token['age'], min=0, max=config.bertconfig.max_age-1)
            if args.model != 'behrt-de':
                token = {k: v for k, v in token.items() if k != 'domain'}
            if args.model == 'medbert':
                token['plos_target'] = torch.Tensor(
                    pickleload(os.path.join(tpath, f'Pretraining_plos_{file}.pkl')))
            dataset = MLM_Dataset(
                token, 
                vocabulary=vocab, 
                masked_ratio=config.datasetconfig.masked_ratio,
                ignore_special_tokens=True,)
            load_dataloader[file] = dataset
        else:
            token = pickleload(os.path.join(tpath, f'Pretraining_tokens_ethos_{file}.pkl'))
            token['age'] = torch.clamp(token['age'], min=0, max=config.bertconfig.max_age-1)
            dataset = EthosDataset(token)
            load_dataloader[file] = dataset

    bertconfig = BertConfig(vocab_size=len(vocab), **config.bertconfig)
    if args.model in ('behrt', 'behrt-de'):
        model = BehrtForMaskedLM(bertconfig)
    elif args.model == 'medbert':
        model = MedbertForMaskedLM(bertconfig)
    elif args.model == 'ethos':
        model = Ethos(bertconfig)
    else:
        assert "Cannot applicable the model"
    
    if args.rep_type != 'none':
        special_tokens_rep_path = os.path.join(rep_path, f'special_tokens_representation.npy')
        if os.path.exists(special_tokens_rep_path):
            special_tokens_rep = np.load(special_tokens_rep_path)
        else:
            special_tokens_rep = torch.randn(7, 768)
            np.save(
                os.path.join(rep_path, f'special_tokens_representation.npy'), 
                special_tokens_rep.numpy())

        ray.init(num_cpus=32)
        desc = mpd.read_csv(os.path.join(dpath, 'all_descriptions.csv'))
        desc['concept_id'] = desc['concept_id'].astype(str)
        desc = desc.set_index('concept_id')
        desc_index = pd.DataFrame(np.arange(desc.shape[0]), index=desc.index, columns=['idx'])
        special_tokens_index = pd.DataFrame(
            np.arange(desc.shape[0], desc.shape[0]+7),
            index=['[PAD]', '[MASK]', '[UNK]', '[CLS]', '[SEP]', '8507', '8532'], columns=['idx']
        )
        desc_index = pd.concat([desc_index, special_tokens_index])
        representation = np.load(os.path.join(rep_path, f'concept_representation_{args.rep_type}.npy'))
        representation = np.concatenate([representation, special_tokens_rep])
        representation = representation[desc_index.loc[list(vocab.keys())]['idx'].values]
        ray.shutdown()
        
        if args.model == 'ethos':
            model.embeddings.concept_embeddings.weight.data = torch.Tensor(representation)
            model.embeddings.concept_embeddings.weight.requires_grad = False
        else:
            model.bert.embeddings.concept_embeddings.weight.data = torch.Tensor(representation)
            model.bert.embeddings.concept_embeddings.weight.requires_grad = False


    if args.model == 'ethos':
        Trainer = EthosTrainer
    else:
        Trainer = EHRTrainer
    Trainer = Trainer(
        model=model,
        train_dataset=load_dataloader['train'],
        valid_dataset=load_dataloader['valid'],
        args=args,
    )

    if args.multi_gpu:
        mp.spawn(Trainer.train, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        Trainer.train(args=args)


if __name__ == '__main__':
    main()

