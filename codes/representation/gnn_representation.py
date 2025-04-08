"""
    Yield BioBERT representations of all concept IDs.
    Input:
        1. data/concepts/CONCEPT_RELATIONSHIP.csv
        2. data/concepts/tmp/final_concepts.csv
        3. usedata/representation/concept_representation_biobert.npy
    Output:
        1. results/gnn/logs/biobert.log
        2. results/gnn/saved/biobert.tar
        3. usedata/representation/concept_representation_biobert+gnn.npy
        
"""

import os
import numpy as np
import modin.pandas as mpd
import ray
import yaml
from yaml import SafeLoader
import random
import argparse
import logging
import time
import datetime

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import dropout_edge

import sys
from pathlib import Path
abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)
from src.graph_model import Model, drop_feature


parser = argparse.ArgumentParser()
parser.add_argument("--num_cpus", type=int, default=32, 
                    help='num cpus') 
parser.add_argument("--rep_type", type=str,
                    help='representation type') 
parser.add_argument("-d", "--device", type=int, default=1, 
                    help='representation type') 
parser.add_argument("--bs", type=int, default=2048, 
                    help='batch size') 
args = parser.parse_args()



if __name__ == '__main__':

    abspath = str(Path(__file__).resolve().parent.parent.parent)
    cpath = os.path.join(abspath, 'data/concepts')
    cpath_tmp = os.path.join(abspath, 'usedata/descriptions')
    rpath = os.path.join(abspath, 'usedata/representation')
    config_path = os.path.join(abspath, 'codes/configs/')
    logpath = os.path.join(abspath, f'results/gnn/logs/{args.rep_type}.log')
    savepath = os.path.join(abspath, f'results/gnn/saved/{args.rep_type}.tar')
    os.makedirs(os.path.join(abspath, f'results/gnn/logs/'), exist_ok=True)
    os.makedirs(os.path.join(abspath, f'results/gnn/saved/'), exist_ok=True)


    # Edge contruction
    edge_path = os.path.join(rpath, 'edges.npy')
    if not os.path.exists(edge_path):
        ray.init(num_cpus=args.num_cpus)
        relationships = mpd.read_csv(os.path.join(cpath,'CONCEPT_RELATIONSHIP.csv'), delimiter='\t')
        concepts = mpd.read_csv(os.path.join(cpath_tmp, "all_descriptions.csv"))

        # Get concept IDs
        use_cid = concepts['concept_id'].apply(lambda x: int(str(x).split('_')[0])).unique()

        # Set graph edges
        relationships = relationships[
            (relationships['concept_id_1'].isin(use_cid)) & 
            (relationships['concept_id_2'].isin(use_cid)) & 
            (relationships['concept_id_1'] > relationships['concept_id_2'])
        ]
        relationships = relationships[['concept_id_1', 'concept_id_2']]

        # Add edges of measurement deciles
        measurement_ids = concepts[
            concepts['concept_id'].apply(lambda x: len(str(x).split('_')) == 2)]
        measurement_ids['concept_id_1'] = measurement_ids['concept_id'].apply(lambda x: int(str(x).split('_')[0]))
        measurement_ids = measurement_ids[['concept_id_1', 'concept_id']]
        measurement_ids.columns = ['concept_id_1', 'concept_id_2']

        # Finalize edges 
        relationships = mpd.concat([relationships, measurement_ids]).drop_duplicates()
        relationships['concept_id_1'] = relationships['concept_id_1'].astype(str)
        relationships['concept_id_2'] = relationships['concept_id_2'].astype(str)

        # Set vocab (index for each concept ID)
        concepts['concept_id'] = concepts['concept_id'].astype(str)
        vocab = concepts.reset_index().set_index('concept_id')['index']

        # Construct torch_geometric Data
        edges = vocab.loc[relationships.values.reshape(-1)].values.reshape(-1, 2).T
        np.save(edge_path, edges)
    else:
        edges = np.load(edge_path)


    # Training configs
    device = f'cuda:{args.device}'
    config_path = os.path.join(config_path, 'grace.yaml')
    config = yaml.load(open(config_path), Loader=SafeLoader)

    torch.manual_seed(config['seed'])
    random.seed(config['seed'])


    # Graph Data construction
    representations = np.load(os.path.join(rpath, f'concept_representation_{args.rep_type}.npy'))
    features = torch.Tensor(representations).to(device)
    edges = torch.Tensor(edges).type(torch.LongTensor).to(device)
    graph_data = Data(x=features, edge_index=edges)
    train_loader = NeighborLoader(
        graph_data,  
        num_neighbors=[30, 20, 10],  
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )


    # Model load (GRACE)
    model = Model(config=config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay'])


    # logger setting
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(logpath)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(config)


    # load previous model
    first_iter = 0
    if os.path.exists(savepath):
        print('Saved file exists.')
        checkpoint = torch.load(savepath, map_location=device)
        first_iter = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])


    prev_time = time.time()
    best_loss = 999999
    patience = 0
    for epoch in range(config['num_epochs']):
        epoch_loss = 0
        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            edge_index_1 = dropout_edge(batch.edge_index, p=config['drop_edge_rate_1'])[0]
            edge_index_2 = dropout_edge(batch.edge_index, p=config['drop_edge_rate_2'])[0]
            x_1 = drop_feature(batch.x, config['drop_feature_rate_1'])
            x_2 = drop_feature(batch.x, config['drop_feature_rate_2'])
            z1 = model(x_1, edge_index_1)
            z2 = model(x_2, edge_index_2)

            loss = model.loss(z1, z2, batch_size=0)
            loss.backward()
            optimizer.step()

            # Determine approximate time left
            batches_done = epoch * len(train_loader) + i + 1
            batches_left = config['num_epochs'] * len(train_loader) - batches_done
            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Train batch loss: %f] ETA: %s"
                % (epoch+1, config['num_epochs'], i+1, len(train_loader), loss.item(), time_left)
            )
            epoch_loss += loss.item()
        
        epoch_loss = epoch_loss / len(train_loader)
        logger.info('Epoch [{}/{}], Train Loss: {:.4f}'.format(
                epoch+1, config['num_epochs'], epoch_loss))
        
        if epoch_loss < best_loss - 0.001:
            best_loss = epoch_loss
            best_model_weights = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, savepath)
            logger.info('Save best model...')
            patience = 0
        else:
            patience += 1
            if patience >= config['patience']:
                break


    # Save representations 
    model.load_state_dict(best_model_weights)
    train_loader = NeighborLoader(
        graph_data,  
        num_neighbors=config['num_neighbors'],  
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    with torch.no_grad():
        model.eval()
        all_representations = []
        for batch in train_loader:
            z = model(batch.x, batch.edge_index)[:len(batch.input_id)]
            all_representations.append(z.cpu().detach())
        
        all_representations = torch.cat(all_representations)
    
    np.save(
        os.path.join(rpath, f'concept_representation_{args.rep_type}+gnn.npy'),
        all_representations.cpu().detach().numpy()
    )





