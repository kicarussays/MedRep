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

torch.set_num_threads(32)

parser = argparse.ArgumentParser()
parser.add_argument("--device", '-d', type=int)
args = parser.parse_args()


def main():
    abspath = str(Path(__file__).resolve().parent.parent.parent)
    cpath_tmp = os.path.join(abspath, 'usedata/descriptions')
    spath = os.path.join(abspath, 'usedata/representation')
    rpath = os.path.join(abspath, 'results/representation')
    lmpath = os.path.join(abspath, 'language_models')
    inputs = pickleload(os.path.join(cpath_tmp, 'tokens.pkl'))

    model_path = os.path.join(lmpath, 'deberta')
    model = DebertaForMaskedLM.from_pretrained(
        os.path.join(rpath, model_path))

    if hasattr(model.deberta.encoder, "rel_embeddings"):
        model.deberta.encoder.rel_embeddings = torch.nn.Embedding(
            2048, model.config.hidden_size)

    ds = InferenceDataset(inputs)
    device = f'cuda:{args.device}'

    load_weight = torch.load(
        os.path.join(rpath, 'deberta.tar'), map_location=device)['model']
    load_weight = {k.split('module.')[1]: v for k, v in load_weight.items()}
    model = model.to(device)
    model.load_state_dict(load_weight)
        
    inference_dl = DataLoader(ds, batch_size=32, shuffle=False, pin_memory=True)

    # Inference
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

        # Concatenate all minibatch CLS representations
    all_representations = torch.cat(cls_representations_list, dim=0)
    np.save(
        os.path.join(spath, 'concept_representation_description.npy'),
        all_representations.cpu().detach().numpy())


if __name__ == '__main__':
    main()
