"""
    Yield representations of descriptions for all concept IDs.
    Input:
        1. usedata/descriptions/tokens.pkl
    Output:
        1. usedata/representation/description_representation.npy
        
"""

import os
import numpy as np
import torch
from transformers import (
    DebertaTokenizer, 
    DataCollatorForLanguageModeling, 
    DebertaForMaskedLM,)
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import AdamW

import logging
import time
import datetime
from pathlib import Path
import sys
import argparse
from sklearn.model_selection import train_test_split
abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)
from src.utils import pickleload
from src.dataset import MLM_Dataset

# argument for multi-gpu
parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=32, help='')
parser.add_argument('--lr', type=float, default=5e-5, help='')
parser.add_argument('--max-epoch', type=int, default=1000, help='')
parser.add_argument('--num-workers', type=int, default=4, help='')
parser.add_argument("--gpu-devices", type=int, nargs='+', default=None, help="")
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world-size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
args = parser.parse_args()


def main():
    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    os.environ["NCCL_TIMEOUT"] = '1200'
    os.environ["MASTER_PORT"] = '12355'
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size


    abspath = str(Path(__file__).resolve().parent.parent.parent)
    cpath_tmp = os.path.join(abspath, 'usedata/descriptions')
    spath = os.path.join(abspath, 'usedata/representation')
    rpath = os.path.join(abspath, 'results/representation')
    lmpath = os.path.join(abspath, 'language_models')
    os.makedirs(rpath, exist_ok=True)

    # Load DeBERTa model and tokenizer
    # We modified max_seq_length 
    print('Model load...')
    model_path = os.path.join(lmpath, 'deberta')
    if os.path.exists(model_path):
        print(f'{model_path}: exist')
        tokenizer = DebertaTokenizer.from_pretrained(model_path)
        model = DebertaForMaskedLM.from_pretrained(model_path)
    else:
        tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
        model = DebertaForMaskedLM.from_pretrained("microsoft/deberta-base")

    if hasattr(model.deberta.encoder, "rel_embeddings"):
        model.deberta.encoder.rel_embeddings = torch.nn.Embedding(
            2048, model.config.hidden_size)
    print('Done')

    # Load tokenized description data
    print('Data load...')
    inputs = pickleload(os.path.join(cpath_tmp, 'tokens.pkl'))
    train_idx, test_idx = train_test_split(
        np.arange(inputs['input_ids'].shape[0]), random_state=50, test_size=0.2)
    train_ds = MLM_Dataset(
        {k: v[train_idx] for k, v in inputs.items()}, 
        input_ids='input_ids',
        target_ids='labels',
        vocabulary=tokenizer.get_vocab()
    )
    test_ds = MLM_Dataset(
        {k: v[test_idx] for k, v in inputs.items()}, 
        input_ids='input_ids',
        target_ids='labels',
        vocabulary=tokenizer.get_vocab()
    )
    print('Done')

    mp.spawn(
        train, 
        nprocs=ngpus_per_node, 
        args=(ngpus_per_node, args, train_ds, test_ds, model, rpath),
    )


def train(
        gpu=None, 
        ngpus_per_node=None, 
        args=None, 
        train_ds=None, 
        test_ds=None, 
        model=None,
        rpath=None):
    args = args
    device = f'cuda:{gpu}'
    ngpus_per_node = torch.cuda.device_count()    
    print("Use GPU: {} for training".format(gpu))

    args.rank = args.rank * ngpus_per_node + gpu    
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    torch.cuda.set_device(gpu)


    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(rpath, 'deberta.log'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


    args.bs = int(args.bs / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[gpu], find_unused_parameters=True)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        eps=1e-8,
    )
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True)
    train_dl = DataLoader(train_ds, batch_size=args.bs, 
                            shuffle=(train_sampler is None), num_workers=args.num_workers, 
                            sampler=train_sampler, pin_memory=True)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(test_ds, shuffle=True)
    valid_dl = DataLoader(test_ds, batch_size=args.bs, 
                            shuffle=(valid_sampler is None), num_workers=args.num_workers, 
                            sampler=valid_sampler, pin_memory=True)

    first_iter = 0
    if os.path.exists(os.path.join(rpath, 'deberta.tar')):
        print('Saved file exists.')
        checkpoint = torch.load(os.path.join(rpath, 'deberta.tar'), map_location=device)
        first_iter = epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    best_loss = 999999
    prev_time = time.time()
    scaler = torch.amp.GradScaler()
    for epoch in range(first_iter, args.max_epoch):
        epoch = epoch
        model.train()
        epoch_loss = 0

        for i, batch in enumerate(train_dl):
            # Train step
            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.float16):
                for key, value in batch.items():
                    batch[key] = value.to(device)
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

            # Determine approximate time left
            batches_done = epoch * len(train_dl) + i + 1
            batches_left = args.max_epoch * len(train_dl) - batches_done
            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Train batch loss: %f] ETA: %s"
                % (epoch+1, args.max_epoch, i+1, len(train_dl), loss.item(), time_left)
            )
            if i > 200: break
        epoch_loss = epoch_loss / 200

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i, batch in enumerate(valid_dl):
                for key, value in batch.items():
                    batch[key] = value.to(device)
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d]"
                    % (epoch+1, args.max_epoch, i+1, len(valid_dl))
                )
                if i > 100: break
        val_loss = val_loss / 100

        valid_flag = False
        if args.rank == 0:
            valid_flag = True
                
        if valid_flag:
            # Print epoch info
            logger.info('Epoch [{}/{}], Train Loss: {:.4f} Val Loss: {:.4f}'.format(
                epoch+1, args.max_epoch, epoch_loss, val_loss))

            if val_loss < best_loss - 0.001:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict() 
                }, os.path.join(rpath, 'deberta.tar'))
                patience = 0
                logger.info('Save best model.')
            
            else:
                patience += 1
                if patience >= 30:
                    logger.info('Early stopping activated.')
                    break


if __name__ == '__main__':
    main()