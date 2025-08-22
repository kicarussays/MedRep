"""
    Yield representations of descriptions for all concept IDs.
    Input:
        1. usedata/descriptions/tokens.pkl
    Output:
        1. usedata/representation/description_representation.npy
        
"""
import argparse
import os
import numpy as np
import torch
from transformers import (
    DebertaForMaskedLM,)
import multiprocessing
import torch
from torch.utils.data import DataLoader
import time
import datetime
from pathlib import Path
import sys
abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)
from src.utils import pickleload
from src.dataset import InferenceDataset
import torch.multiprocessing as mp

torch.set_num_threads(32)

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


def run_inference(rank, model, data_chunks):
    abspath = str(Path(__file__).resolve().parent.parent.parent)
    spath = os.path.join(abspath, 'usedata/representation')

    print("Use GPU: {} for training".format(rank))
    device = torch.device(f'cuda:{rank}')
    model.to(device)

    ds = InferenceDataset(data_chunks[rank])
    inference_dl = DataLoader(ds, batch_size=16, shuffle=False, pin_memory=True)

    with torch.no_grad():
        model.eval()
        cls_representations_list = []
        prev_time = time.time()
        for i, batch in enumerate(inference_dl):
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            output = model(**batch, output_hidden_states=True)
            cls_representations = output.hidden_states[-1][:, 0, :].cpu().detach()
            cls_representations_list.append(cls_representations)

            # Determine approximate time left
            batches_done = i + 1
            batches_left = len(inference_dl) - batches_done
            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            sys.stdout.write(
                "\r[Batch %d/%d] ETA: %s"
                % (i+1, len(inference_dl), time_left)
            )
    all_representations = torch.cat(cls_representations_list, dim=0)        
    np.save(
        os.path.join(spath, f'concept_representation_description_{rank}.npy'),
        all_representations.cpu().detach().numpy())



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
    model_path = os.path.join(lmpath, 'deberta')

    model = DebertaForMaskedLM.from_pretrained(
        os.path.join(rpath, model_path))
    if hasattr(model.deberta.encoder, "rel_embeddings"):
        model.deberta.encoder.rel_embeddings = torch.nn.Embedding(
            2048, model.config.hidden_size)

    device = f'cuda:{args.gpu_devices[0]}'
    load_weight = torch.load(
        os.path.join(rpath, 'deberta.tar'), map_location=device)['model']
    load_weight = {k.split('module.')[1]: v for k, v in load_weight.items()}
    model = model.to(device)
    model.load_state_dict(load_weight)

    inputs = pickleload(os.path.join(cpath_tmp, 'tokens.pkl'))
    chunks = np.array_split(list(range(inputs['input_ids'].shape[0])), ngpus_per_node)
    data_chunks = {}
    for idx, chunk in enumerate(chunks):
        data_chunks[idx] = {k: v[chunk] for k, v in inputs.items()}
        
    mp.spawn(
        run_inference, 
        nprocs=ngpus_per_node, 
        args=(model, data_chunks),
    )


if __name__ == '__main__':
    main()
