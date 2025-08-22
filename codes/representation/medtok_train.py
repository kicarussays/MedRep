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
from tqdm import tqdm

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
from medtok.tokenizer import MultimodalTokenizer
from medtok.loss import shared_loss, specific_loss



abspath = str(Path(__file__).resolve().parent.parent.parent)
cpath = os.path.join(abspath, 'data/concepts')
cpath_tmp = os.path.join(abspath, 'usedata/descriptions')
rpath = os.path.join(abspath, 'usedata/representation')
config_path = os.path.join(abspath, 'codes/configs/')
logpath = os.path.join(abspath, f'results/gnn/logs/medtok.log')
savepath = os.path.join(abspath, 'results/gnn/saved/medtok.tar')
os.makedirs(os.path.join(abspath, f'results/gnn/logs/'), exist_ok=True)
os.makedirs(os.path.join(abspath, f'results/gnn/saved/'), exist_ok=True)


def main(args):


    abspath = str(Path(__file__).resolve().parent.parent.parent)
    cpath = os.path.join(abspath, 'data/concepts')
    cpath_tmp = os.path.join(abspath, 'usedata/descriptions')
    rpath = os.path.join(abspath, 'usedata/representation')
    config_path = os.path.join(abspath, 'codes/configs/')
    logpath = os.path.join(abspath, f'results/gnn/logs/medrep.log')
    savepath = os.path.join(abspath, f'results/gnn/saved/medrep.tar')
    os.makedirs(os.path.join(abspath, f'results/gnn/logs/'), exist_ok=True)
    os.makedirs(os.path.join(abspath, f'results/gnn/saved/'), exist_ok=True)


    # Edge contruction
    edge_path = os.path.join(rpath, 'edges.npy')
    if not os.path.exists(edge_path):
        ray.init(num_cpus=args.num_cpus)
        relationships = mpd.read_csv(os.path.join(cpath,'CONCEPT_RELATIONSHIP.csv'), delimiter='\t')
        concepts = mpd.read_csv(os.path.join(rpath, "concept_idx.csv"))

        # Get concept IDs
        use_cid = concepts['concept_id'].values

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
    representations = np.load(os.path.join(rpath, f'concept_representation_description.npy'))
    features = torch.Tensor(representations)
    edges = torch.Tensor(edges).type(torch.LongTensor)
    graph_data = Data(x=features.contiguous(), edge_index=edges.contiguous())
    train_loader = NeighborLoader(
        graph_data,  
        num_neighbors=[30, 20, 10],  
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )

    vq_model = MultimodalTokenizer(
        graph_model_name='GCN',
        graph_in_channels=192,
        graph_hidden_channels=384,
        graph_out_channels=192,
        codebook_size=30000,
        codebook_embed_dim=192,  ##codebook for graph
        #semantic_code_dim=args.semantic_code_dim,    ##codebook for text
        commit_loss_beta=0.25,
        entropy_loss_ratio=0.0,
        #dropout_p=args.dropout_p,
        #kmeans=args.kmeans,
    )
    vq_model = vq_model.to(device)
    scaler = torch.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    scaler_disc = torch.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))


    # logger setting
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(logpath)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


    logger.info("Optimizing all parameters.")
    logger.info(f"no kmeans, args.lr = {args.lr}")
    optimizer = torch.optim.Adam(vq_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    args.vq_ckpt = savepath
    if os.path.exists(savepath):
        load_weight = torch.load(savepath, map_location=device, weights_only=False)
        vq_model.load_state_dict(load_weight['model'])
        optimizer.load_state_dict(load_weight['optimizer'])

    # Prepare models for training:
    train_steps = load_weight['steps'] if os.path.exists(savepath) else 0 
    start_epoch = 0
    vq_model.train()


    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")

    stop_flag = False
    for epoch in range(start_epoch, args.epochs):
        
        logger.info(f"Beginning epoch {epoch}...")
        for inputs in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}"):
            inputs.x = inputs.x.to(device)
            inputs.edge_index = inputs.edge_index.to(device)
            inputs.n_id = inputs.n_id.to(device)
            inputs.e_id = inputs.e_id.to(device)
            inputs.input_id = inputs.input_id.to(device)

            # generator training
            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=ptdtype):  
                quantized_result = vq_model(inputs)
                #loss = codebook_loss[0] + codebook_loss[1] + codebook_loss[2] + codebook_loss[3]
                codebook_loss = (quantized_result['shared_embed_loss'][0] + quantized_result['shared_embed_loss'][1] +
                                quantized_result['text_specific_loss'][0] + quantized_result['text_specific_loss'][1] + 
                                quantized_result['graph_specific_loss'][0]+quantized_result['graph_specific_loss'][1])

                shared_loss_all = shared_loss(quantized_result['shared_text_embedding'], quantized_result['shared_graph_embedding'], quantized_result['text_feature'], quantized_result['graph_feature'])


                specific_loss_all = specific_loss(z1 = quantized_result['specific_embedding_text'],
                                            z1_aug = quantized_result['specific_embedding_text'],
                                            z2 = quantized_result['specific_embedding_graph'],
                                            z2_aug = quantized_result['specific_embedding_graph'],
                                            z1_c = quantized_result['shared_text_embedding'],
                                            z2_c = quantized_result['shared_graph_embedding'])
                
                beta = args.commit_loss_beta
                lamb = args.specific_loss_lamb
                loss_common = codebook_loss + beta * shared_loss_all + lamb * specific_loss_all
        
            scaler.scale(loss_common).backward()

            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vq_model.parameters(), args.max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            
            '''with torch.cuda.amp.autocast(dtype=ptdtype):
                quantized_result_star = vq_model(inputs)

                specific_loss_all = specific_loss(z1 = quantized_result['specific_embedding_text'],
                                              z1_aug = quantized_result_star['specific_embedding_text_aug'],
                                              z2 = quantized_result['specific_embedding_graph'],
                                              z2_aug = quantized_result_star['specific_embedding_graph_aug'],
                                              z1_c = quantized_result['shared_text_embedding'],
                                              z2_c = quantized_result['shared_graph_embedding'])
                loss_specific = specific_loss_all   
            optimizer.zero_grad()
            scaler.scale(loss_specific).backward()
            scaler.step(optimizer)
            scaler.update()'''

            # # Log loss values:
            loss = loss_common.item() #+ loss_specific.item()
            running_loss += loss
            
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                
                #codebook_loss = codebook_loss.item()
                #print(codebook_loss)

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if train_steps == 1500:
                model_weight = vq_model.state_dict()  
                checkpoint = {
                    "model": model_weight,
                    "optimizer": optimizer.state_dict(),
                    "steps": train_steps,
                    "args": args
                }
                
                torch.save(checkpoint, savepath)    
    
                train_loader = NeighborLoader(
                    graph_data,  
                    num_neighbors=[30, 20, 10],  
                    batch_size=2048, 
                    shuffle=False, 
                    num_workers=4,
                    pin_memory=True
                )
                with torch.no_grad():
                    vq_model.eval()
                    all_reps = []
                    for inputs in tqdm(train_loader): 
                        inputs.x = inputs.x.to(device)
                        inputs.edge_index = inputs.edge_index.to(device)
                        inputs.n_id = inputs.n_id.to(device)
                        inputs.e_id = inputs.e_id.to(device)
                        inputs.input_id = inputs.input_id.to(device)
                        quantized_result = vq_model(inputs)
                        reps = torch.cat([
                            quantized_result['specific_embedding_text'],
                            quantized_result['specific_embedding_graph'],
                            quantized_result['shared_text_embedding'],
                            quantized_result['shared_graph_embedding'],
                        ], dim=-1).cpu().detach()
                        all_reps.append(reps)
                    all_reps = torch.cat(all_reps)
                
                np.save(
                    os.path.join(rpath, f'concept_representation_medtok_{train_steps}.npy'),
                    all_reps.cpu().detach().numpy()
                )
                vq_model.train()
            if stop_flag: break
        if stop_flag: break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='datasets/')
    parser.add_argument("--device", type=int, default=6)

    parser.add_argument("--kg-path", type=str, default='/n/netscratch/mzitnik_lab/Lab/xsu/primeKG/', help="path to the knowledge graph")
    parser.add_argument("--med-codes-pkg-map-path", type=str, default='/n/holylfs06/LABS/mzitnik_lab/Lab/shvat372/icml_paper/ICML_codes/graphs/all_codes_mappings_v3.parquet', help="path to the med codes package map")
    parser.add_argument("--graph-save-path", type=str, default='/n/netscratch/mzitnik_lab/Lab/xsu/kg_temp_2912', help="path to save the graph")
    
    parser.add_argument("--data-face-path", type=str, default=None, help="face datasets to improve vq model")
    parser.add_argument("--cloud-save-path", type=str, default='/n/netscratch/mzitnik_lab/Lab/xsu/MultimodalTokenizer/log/', help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--model", type=str, default="MultimodalTokenizer")
    parser.add_argument("--graph_model_name", type=str, choices=["GCN", "GAT", "GraphTransformer"], default="GCN")
    parser.add_argument("--text_model_name", type=str, choices=["bert-base-uncased"], default="bert-base-uncased")

    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--finetune", action='store_true', help="finetune a pre-trained vq model")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--graph_in_channels", type=int, default=64, help="input channels for graph encoder")
    parser.add_argument("--graph_hidden_channels", type=int, default=128, help="hidden channels for graph encoder") 
    parser.add_argument("--graph_out_channels", type=int, default=64, help="output channels for graph encoder")

    parser.add_argument("--codebook-size", type=int, default=30000, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=64, help="codebook dimension for graph quantization")
    parser.add_argument("--semantic-code-dim", type=int, default=64, help="codebook dimension for semantic quantization")
    parser.add_argument("--text-code-dim", type=int, default=64, help="codebook dimension for semantic quantization")
    parser.add_argument("--codebook-l2-norm", action='store_true', default=True, help="l2 norm codebook")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    parser.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    parser.add_argument("--shared-loss-beta", type=float, default=0.5, help="shared loss beta in codebook loss")
    parser.add_argument("--specific-loss-lamb", type=float, default=0.5, help="specific loss lambda in codebook loss")

    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--dropout-p", type=float, default=0.2, help="dropout_p")
    parser.add_argument("--results-dir", type=str, default="/n/netscratch/mzitnik_lab/Lab/xsu/MultimodalTokenizer/pre_trained_model")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=1024) 
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-checkpoints", type=int, default=2)
    parser.add_argument("--mixed-precision", type=str, default='fp16', choices=["none", "fp16", "bf16"]) # better change to bf16 if GPU support

    parser.add_argument("--infer_interpolate", action='store_true', help="interpolate the positional encoding for higher resolution inference")
    parser.add_argument("--enhanced_decoder", action='store_true', help="whether using enhanced decoder")
    parser.add_argument("--kmeans", action='store_true', help="whether using kmeans for codebook initialization")
    parser.add_argument('--finetune_decoder', action='store_true', help='finetune decoder')
    parser.add_argument("--num-cpus", type=int, default=64)
    args = parser.parse_args()
    main(args)