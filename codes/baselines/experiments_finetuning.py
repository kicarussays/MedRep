import os
import numpy as np
import pandas as pd
import torch
import argparse
import yaml
from transformers import BertConfig
import ray
import modin.pandas as mpd
from torch.utils.data import DataLoader
torch.set_num_threads(8)

parser = argparse.ArgumentParser()
parser.add_argument('--device', '-d', type=str, 
                    help='cpu or GPU Device Number', default=0)
parser.add_argument('--seed', type=int, 
                    help='Random seed', default=100)
parser.add_argument('--hospital', type=str, 
                    help='hospital', default='mimic')
parser.add_argument('--model', type=str, 
                    help='retain, behrt, behrt-de, medbert, ethos', default='behrt-de')
parser.add_argument('--outcome', type=str, 
                    help='MT, LLOS, or RA', default='MT')
parser.add_argument('--rep-type', type=str,
                    help='Select representation type', default='none')
parser.add_argument('--smooth', action='store_true',
                    help='loss smoothing option')
parser.add_argument('--aug', action='store_true',
                    help='augmentation option')
parser.add_argument('--aug-times', type=int, 
                    help='How many times to augment', default=5)
args = parser.parse_args()


import sys
from pathlib import Path
abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)
from src.utils import seedset, pickleload, DotDict
from src.dataset import BinaryOutcomeDataset
from src.trainer import EHRClassifier
from src.baseline_models import (
    RNNForSequenceClassification, 
    EHRBertForSequenceClassification, 
    EthosForSequenceClassification)
configpath = os.path.join(abspath, f'configs/{args.model}.yaml')

abspath = str(Path(__file__).resolve().parent.parent.parent)
spath = os.path.join(abspath, 'usedata/{hospital}/')
tpath = os.path.join(abspath, 'usedata/{hospital}/tokens/Finetuning')
rpath = os.path.join(abspath, f'results/{args.hospital}/finetuning/')
dpath = os.path.join(abspath, f'usedata/descriptions/')
ppath = os.path.join(abspath, f'results/{args.hospital}/pretraining/saved/')
rep_path = os.path.join(abspath, f'usedata/representation')
lpath = os.path.join(abspath, f'results/logits/')
os.makedirs(lpath, exist_ok=True)


def main():
    with open(configpath, 'r') as f: 
        config = yaml.safe_load(f)
        args.bs = config['trainconfig']['bs']
        args.lr = config['trainconfig']['lr']
        args.max_epoch = config['trainconfig']['max_epoch']
        config['bertconfig']['smooth'] = args.smooth
        config['bertconfig']['rep_type'] = args.rep_type
        config = DotDict(**config)
        
    device = f'cuda:{args.device}'
    seedset(args.seed)
    os.makedirs(os.path.join(rpath, f'logs'), exist_ok=True)
    os.makedirs(os.path.join(rpath, f'saved'), exist_ok=True)
    if args.aug:
        param_option = f'rep_type={args.rep_type}_smooth={args.smooth}_aug={args.aug_times}'
    else:
        param_option = f'rep_type={args.rep_type}_smooth={args.smooth}_aug={args.aug}'

    pretrained_param = f'rep_type={args.rep_type}_smooth={args.smooth}'
    args.logpth = os.path.join(rpath, f'logs/{args.model}+{param_option}_{args.outcome}.log')
    args.savepth = os.path.join(rpath, f'saved/{args.model}+{param_option}_{args.outcome}.tar')
    
    vocab = torch.load(os.path.join(spath.format(hospital=args.hospital), 'vocab.pt'))
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
        special_tokens = ['[PAD]', '[MASK]', '[UNK]', '[CLS]', '[SEP]', '8507', '8532']
        special_tokens_index = pd.DataFrame(
            np.arange(desc.shape[0], desc.shape[0]+7),
            index=special_tokens, columns=['idx']
        )
        desc_index = pd.concat([desc_index, special_tokens_index])
        representation = np.load(os.path.join(rep_path, f'concept_representation_{args.rep_type}.npy'))
        representation = np.concatenate([representation, special_tokens_rep])

        if args.aug:
            included_desc = pd.read_csv(os.path.join(dpath, 'included_descriptions.csv'))
            included_desc = included_desc.set_index('index')
            neighbors = np.load(os.path.join(rep_path, f'neighbors_{args.rep_type}.npy'))

            # Expand vocabulary
            vocab_keys = list(vocab.keys())
            start_idx = len(vocab)
            for c in included_desc['concept_id']:
                if c not in vocab_keys:
                    vocab[c] = start_idx
                    start_idx += 1
            selected_representation = representation[desc_index.loc[list(vocab.keys())]['idx'].values]

            # Get neighbors indexes
            neighbors_mapped_path = os.path.join(rep_path, f'neighbors_mapped_vocab_{args.hospital}_{args.rep_type}.npy')
            if os.path.exists(neighbors_mapped_path):
                neighbors_mapped = np.load(neighbors_mapped_path)
            else:
                vocab_df = pd.DataFrame(vocab.values(), index=vocab.keys(), columns=['vocab_idx'])
                neighbor_mapping = pd.DataFrame(
                    vocab_df.loc[included_desc['concept_id']]['vocab_idx'].values, 
                    index=included_desc['idx'],
                    columns=['vocab_idx'])
                special_tokens_df = vocab_df.loc[special_tokens]
                special_tokens_df.index = np.arange(
                    neighbor_mapping.shape[0], 
                    neighbor_mapping.shape[0]+len(special_tokens))
                neighbor_mapping = pd.concat([
                    neighbor_mapping, special_tokens_df
                ])

                neighbors_mapped = np.vectorize(lambda x: neighbor_mapping.loc[x])(neighbors)
                neighbors_mapped = np.concatenate([
                    neighbors_mapped,
                    np.repeat(vocab_df.loc[special_tokens]['vocab_idx'].values, 30).reshape(-1, 30)
                ])
                neighbors_mapped = neighbors_mapped[neighbor_mapping.sort_values(['vocab_idx']).index]
                np.save(neighbors_mapped_path, neighbors_mapped)
        else:
            selected_representation = representation[desc_index.loc[list(vocab.keys())]['idx'].values]
        ray.shutdown()

    load_dataloader = {}
    for file in ('train', 'valid'):
        token, label = pickleload(os.path.join(
            tpath.format(hospital=args.hospital), 
            f'{args.outcome}_{args.hospital}_from_vocab_{args.hospital}_{file}.pkl'))
        token['age'] = torch.clamp(token['age'], min=0, max=config.bertconfig.max_age-1)
        label = torch.Tensor(label).type(torch.LongTensor)
        if args.model != 'behrt-de':
            token = {k: v for k, v in token.items() if k != 'domain'}
        dataset = BinaryOutcomeDataset(
            token, 
            outcomes=label,
            vocabulary=vocab, 
            ignore_special_tokens=True,
            neighbors_mapped=neighbors_mapped if args.aug and file == 'train' else None,
            aug_times=args.aug_times if args.aug and file == 'train' else None
        )
        load_dataloader[file] = dataset

    bertconfig = BertConfig(
        vocab_size=len(vocab), 
        problem_type='single_label_classification',
        **config.bertconfig)
    if args.model in ('retain', ):
        model = RNNForSequenceClassification(bertconfig, args.model)
    elif args.model in ('behrt', 'medbert', 'behrt-de'):
        model = EHRBertForSequenceClassification(bertconfig)
        load_pretrained = torch.load(os.path.join(ppath, f'{args.model}+{pretrained_param}.tar'),
                                    map_location=device)
        model.load_state_dict(load_pretrained['model'], strict=False)
    elif args.model in ('ethos', ):
        model = EthosForSequenceClassification(bertconfig)
        load_pretrained = torch.load(os.path.join(ppath, f'{args.model}+{pretrained_param}.tar'),
                                    map_location=device)
        model.load_state_dict(load_pretrained['model'], strict=False)

    if args.rep_type != 'none':
        if args.model in ('retain', 'ethos'):
            model.embeddings.concept_embeddings.weight.data = torch.Tensor(selected_representation)
            model.embeddings.concept_embeddings.weight.requires_grad = False
        else:
            model.bert.embeddings.concept_embeddings.weight.data = torch.Tensor(selected_representation)
            model.bert.embeddings.concept_embeddings.weight.requires_grad = False
    model = model.to(device)

    
    # Training classifier 
    Trainer = EHRClassifier(
        model=model,
        train_dataset=load_dataloader['train'],
        valid_dataset=load_dataloader['valid'],
        args=args
    )
    Trainer.train()
    checkpoint = torch.load(args.savepth, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    # Testing
    all_hosp = ['mimic', 'ehrshot']
    for hosp in all_hosp:
        logit_path = os.path.join(lpath, f'{args.model}+{param_option}_{args.outcome}_model_mimic_data_{hosp}.pkl')
        
        if args.rep_type == 'none':
            token, label = pickleload(os.path.join(
                tpath.format(hospital=hosp), 
                f'{args.outcome}_{hosp}_from_vocab_mimic_test.pkl'))
        else:
            token, label = pickleload(os.path.join(
                tpath.format(hospital=hosp), 
                f'{args.outcome}_{hosp}_from_vocab_{hosp}_test.pkl'))
            vocab = torch.load(os.path.join(spath.format(hospital=hosp), 'vocab.pt'))
            selected_representation = representation[desc_index.loc[list(vocab.keys())]['idx'].values]
            if args.model in ('retain', 'ethos'):
                model.embeddings.concept_embeddings.weight.data = torch.Tensor(selected_representation)
                model.embeddings.concept_embeddings.weight.requires_grad = False
            else:
                model.bert.embeddings.concept_embeddings.weight.data = torch.Tensor(selected_representation)
                model.bert.embeddings.concept_embeddings.weight.requires_grad = False
            model = model.to(device)

        label = torch.Tensor(label).type(torch.LongTensor)
        if args.model != 'behrt-de':
            token = {k: v for k, v in token.items() if k != 'domain'}
        dataset = BinaryOutcomeDataset(
            token, 
            outcomes=label,
            vocabulary=vocab, 
            ignore_special_tokens=True,
            neighbors_mapped=None,
            aug_times=None
        )
        test_dl = DataLoader(dataset, batch_size=256, shuffle=False)
        Trainer = EHRClassifier(
            model=model,
            train_dataset=load_dataloader['train'], # Redundant
            valid_dataset=load_dataloader['valid'], # Redundant
            args=args
        )
        Trainer.save_logits(test_dl, logit_path)


if __name__ == '__main__':
    main()

