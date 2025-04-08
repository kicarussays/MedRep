import numpy as np
import os
import sys
from tqdm import tqdm
import time
import datetime
import logging
import traceback
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
from torch.optim import AdamW
import torch.distributed as dist

import sys
from pathlib import Path
abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)

from src.stat_function import (
    auc_ci, prc_ci, calculate_confidence_interval,
    calculate_sensitivity, calculate_specificity, 
    calculate_precision, calculate_f1_score
)
from src.utils import r3, picklesave


class EHRTrainer:
    def __init__(self, 
        model: torch.nn.Module,
        train_dataset: Dataset = None,
        valid_dataset: Dataset = None,
        args: dict = {},
    ):
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=0.01,
            eps=1e-8,
        )
        self.args = args

    def train(self, gpu=None, ngpus_per_node=None, args=None):
        if self.args.multi_gpu:
            self.args = args
            self.device = f'cuda:{gpu}'
            ngpus_per_node = torch.cuda.device_count()    
            print("Use GPU: {} for training".format(gpu))
        
            self.args.rank = self.args.rank * ngpus_per_node + gpu    
            dist.init_process_group(backend=self.args.dist_backend, init_method=self.args.dist_url,
                                    world_size=self.args.world_size, rank=self.args.rank)
            
            torch.cuda.set_device(gpu)
            self.model.to(self.device)
            self.args.bs = int(self.args.bs / ngpus_per_node)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[gpu], find_unused_parameters=True)
        

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.args.logpth)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        first_iter = 0
        if os.path.exists(self.args.savepth):
            print('Saved file exists.')
            checkpoint = torch.load(self.args.savepth, map_location=self.device)
            first_iter = self.epoch = checkpoint["epoch"] + 1
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            
        if self.args.multi_gpu:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
            self.train_dl = DataLoader(self.train_dataset, batch_size=self.args.bs, 
                                    shuffle=(train_sampler is None), num_workers=self.args.num_workers, 
                                    sampler=train_sampler, pin_memory=True)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_dataset, shuffle=False)
            self.valid_dl = DataLoader(self.valid_dataset, batch_size=self.args.bs, 
                                    shuffle=(valid_sampler is None), num_workers=self.args.num_workers, 
                                    sampler=valid_sampler, pin_memory=True)
        else:
            self.train_dl = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True)
            self.valid_dl = DataLoader(self.valid_dataset, batch_size=self.args.bs, shuffle=False)

        best_loss = 999999
        prev_time = time.time()
        scaler = torch.amp.GradScaler()
        try:
            for epoch in range(first_iter, self.args.max_epoch):
                self.epoch = epoch
                self.model.train()
                epoch_loss = 0

                for i, batch in enumerate(self.train_dl):
                    # Train step
                    self.optimizer.zero_grad()
                    with torch.autocast(device_type=self.device, dtype=torch.float16):
                        outputs = self.forward_pass(batch)
                        loss = outputs.loss

                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    epoch_loss += loss.item()

                    # Determine approximate time left
                    batches_done = epoch * len(self.train_dl) + i + 1
                    batches_left = self.args.max_epoch * len(self.train_dl) - batches_done
                    time_left = datetime.timedelta(
                        seconds=batches_left * (time.time() - prev_time))
                    prev_time = time.time()

                    sys.stdout.write(
                        "\r[Epoch %d/%d] [Batch %d/%d] [Train batch loss: %f] ETA: %s"
                        % (epoch+1, self.args.max_epoch, i+1, len(self.train_dl), loss.item(), time_left)
                    )

                # Validate (returns None if no validation set is provided)
                self.model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for batch in self.valid_dl:
                        outputs = self.forward_pass(batch)
                        val_loss += outputs.loss.item()
                    
                self.val_loss = val_loss / len(self.valid_dl)

                valid_flag = False
                if not self.args.multi_gpu:
                    valid_flag = True
                elif self.args.rank == 0:
                    valid_flag = True
                        
                if valid_flag:
                    # Print epoch info
                    self.logger.info('Epoch [{}/{}], Train Loss: {:.4f} Val Loss: {:.4f}'.format(
                        epoch+1, self.args.max_epoch, epoch_loss / len(self.train_dl), self.val_loss))

                    if self.val_loss < best_loss:
                        self.logger.info('Save best model...')
                        self.save_model()
                        best_loss = self.val_loss


            save_flag = False
            if not self.args.multi_gpu:
                save_flag = True
            else:
                if self.args.rank == 0:
                    save_flag = True
        
        except KeyboardInterrupt:
            import sys
            sys.exit('KeyboardInterrupt')

        except:
            logging.error(traceback.format_exc())
        
        if save_flag: self.save_model()


    def forward_pass(self, batch: dict):
        self.to_device(batch)
        model_input = {
            'input_ids': batch['concept'],
            'attention_mask': batch['attention_mask'] if 'attention_mask' in batch else None,
            'age_ids': batch['age'] if 'age' in batch else None,
            'segment_ids': batch['segment'] if 'segment' in batch else None,
            'record_rank_ids': batch['record_rank'] if 'record_rank' in batch else None,
            'domain_ids': batch['domain'] if 'domain' in batch else None,
            'target': batch['target'] if 'target' in batch else None,
        }
        if 'plos_target' in batch:
            model_input['plos_target'] = batch['plos_target']
        return self.model(**model_input)
        

    def to_device(self, batch: dict) -> None:
        """Moves a batch to the device in-place"""
        for key, value in batch.items():
            batch[key] = value.to(self.device)
            
    
    def save_model(self):
        torch.save({
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, self.args.savepth)
            
            
class EHRClassifier(EHRTrainer):
    def __init__(self, 
        model: torch.nn.Module,
        train_dataset: Dataset = None,
        valid_dataset: Dataset = None,
        args: dict = {},
    ):
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=0.01,
            eps=1e-8,
        )
        self.softmax = torch.nn.Softmax(dim=1)
        self.args = args

    def train(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.args.logpth)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        first_iter = 0
        # if os.path.exists(self.args.savepth):
        #     print('Saved file exists.')
        #     checkpoint = torch.load(self.args.savepth, map_location=self.device)
        #     first_iter = checkpoint["epoch"] + 1
        #     self.model.load_state_dict(checkpoint["model"], strict=False)
            # self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        # For 10 times oversampling
        label = self.train_dataset.outcomes
        if label.sum() / len(label) < 0.1:
            class_sample_count = torch.Tensor([(len(label)-label.sum())/10, label.sum()])
            weight = 1. / class_sample_count.float()
            samples_weight = torch.tensor([weight[t] for t in label.type(torch.long)])

            sampler = WeightedRandomSampler(
                weights=samples_weight,
                num_samples=len(samples_weight),
                replacement=True  
            )
            self.train_dl = DataLoader(self.train_dataset, batch_size=self.args.bs, sampler=sampler)
        else:
            self.train_dl = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True)
        self.valid_dl = DataLoader(self.valid_dataset, batch_size=256, shuffle=True)

        best_score = 0
        patience = 0
        prev_time = time.time()
        try:
            import sys
            for epoch in range(first_iter, self.args.max_epoch):
                stop_flag = False
                self.epoch = epoch
                epoch_loss = 0
                for i, batch in enumerate(self.train_dl):
                    self.model.train()
                    # Train step
                    outputs = self.forward_pass(batch)
                    loss = outputs.loss
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    epoch_loss += loss.item()

                    # Determine approximate time left
                    batches_done = epoch * len(self.train_dl) + i + 1
                    batches_left = self.args.max_epoch * len(self.train_dl) - batches_done
                    time_left = datetime.timedelta(
                        seconds=batches_left * (time.time() - prev_time))
                    prev_time = time.time()

                    sys.stdout.write(
                        "\r[Epoch %d/%d] [Batch %d/%d] [Train batch loss: %f] ETA: %s"
                        % (epoch+1, self.args.max_epoch, i+1, len(self.train_dl), loss.item(), time_left)
                    )

                    if (i + 1) % int(len(self.train_dl) / 5) == 0:
                        self.validate(self.valid_dl, partition=False)
                        
                        # Print epoch info
                        self.logger.info('Epoch [{}/{}], Batch [{}/{}], Train Loss: {:.4f} Val Loss: {:.4f} Val AUROC: {:.4f}'.format(
                            epoch+1, self.args.max_epoch, i+1, len(self.train_dl), epoch_loss / len(self.train_dl), self.val_loss, self.roc_auc))

                        if self.roc_auc > best_score + 0.001:
                            self.logger.info('Save best model...')
                            self.save_model()
                            best_score = self.roc_auc
                            patience = 0
                        else:
                            patience += 1
                            if patience >= 10: 
                                self.logger.info('Early stopping activated.')
                                stop_flag = True; break
                if stop_flag: break

        except KeyboardInterrupt:
            import sys
            sys.exit('KeyboardInterrupt')

        except:
            logging.error(traceback.format_exc())


    def validate(self, dl, partition=True):
        with torch.no_grad():
            self.model.eval()
            val_loss = 0
            val_logits = []
            val_labels = []
            n = 0
            for batch in tqdm(dl):
                outputs = self.forward_pass(batch)
                val_loss += outputs.loss.item()
                val_logits.append(self.softmax(outputs.logits)[:, 1])
                val_labels.append(batch['target'])
                if partition and n > len(dl) / 5:
                    break
                n += 1
            
            self.val_loss = val_loss / (n+1)
            self.val_logits = torch.cat(val_logits).cpu().detach().numpy()
            self.val_labels = torch.cat(val_labels).cpu().detach().numpy()

            fpr, tpr, _ = roc_curve(self.val_labels, self.val_logits)
            self.roc_auc = auc(fpr, tpr)
        

    def save_logits(self, dl, path):
        self.validate(dl, partition=False)
        picklesave((self.val_logits, self.val_labels), path)
            
            
class EHRClassifierWithGlobalModel(EHRTrainer):
    def __init__(self, 
        local_model: torch.nn.Module,
        global_model: torch.nn.Module,
        train_dataset: Dataset = None,
        valid_dataset: Dataset = None,
        args: dict = {},
    ):
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        self.local_model = local_model.to(self.device)
        for param in self.local_model.parameters(): param.requires_grad = False
        
        self.global_model = global_model.to(self.device)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.optimizer = AdamW(
            self.global_model.parameters(),
            lr=args.lr,
            weight_decay=0.01,
            eps=1e-8,
        )
        self.softmax = torch.nn.Softmax(dim=1)
        self.args = args

    def train(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.args.logpth)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info('\n\nGlobal Trainer Start...\n\n')

        first_iter = 0
        
        # For 10 times oversampling
        label = self.train_dataset.outcomes
        if label.sum() / len(label) < 0.1:
            class_sample_count = torch.Tensor([(len(label)-label.sum())/10, label.sum()])
            weight = 1. / class_sample_count.float()
            samples_weight = torch.tensor([weight[t] for t in label.type(torch.long)])

            sampler = WeightedRandomSampler(
                weights=samples_weight,
                num_samples=len(samples_weight),
                replacement=True  
    )
            self.train_dl = DataLoader(self.train_dataset, batch_size=self.args.bs, sampler=sampler)
        else:
            self.train_dl = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True)
        self.valid_dl = DataLoader(self.valid_dataset, batch_size=32, shuffle=True)

        best_score = 0
        patience = 0
        prev_time = time.time()
        try:
            import sys
            for epoch in range(first_iter, self.args.max_epoch):
                stop_flag = False
                self.epoch = epoch
                epoch_loss = 0
                for i, batch in enumerate(self.train_dl):
                    self.global_model.train()
                    # Train step
                    dropped_input = self.forward_pass_with_global_model(batch)
                    outputs = self.global_model(**dropped_input)
                    loss = outputs.loss
                    loss.backward()
                    if (i+1) % 10 == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    epoch_loss += loss.item()

                    # Determine approximate time left
                    batches_done = epoch * len(self.train_dl) + i + 1
                    batches_left = self.args.max_epoch * len(self.train_dl) - batches_done
                    time_left = datetime.timedelta(
                        seconds=batches_left * (time.time() - prev_time))
                    prev_time = time.time()

                    sys.stdout.write(
                        "\r[Epoch %d/%d] [Batch %d/%d] [Train batch loss: %f] ETA: %s"
                        % (epoch+1, self.args.max_epoch, i+1, len(self.train_dl), loss.item(), time_left)
                    )

                    if (i + 1) % int(len(self.train_dl) / 5) == 0:
                        self.validate(self.valid_dl)
                        
                        # Print epoch info
                        self.logger.info('Epoch [{}/{}], Batch [{}/{}], Train Loss: {:.4f} Val Loss: {:.4f} Val AUROC: {:.4f}'.format(
                            epoch+1, self.args.max_epoch, i+1, len(self.train_dl), epoch_loss / len(self.train_dl), self.val_loss, self.roc_auc))

                        if self.roc_auc > best_score + 0.001:
                            self.logger.info('Save best model...')
                            self.save_model()
                            best_score = self.roc_auc
                            patience = 0
                        else:
                            patience += 1
                            if patience >= 25: 
                                self.logger.info('Early stopping activated.')
                                stop_flag = True; break
                if stop_flag: break

        except KeyboardInterrupt:
            import sys
            sys.exit('KeyboardInterrupt')

        except:
            logging.error(traceback.format_exc())


    def forward_pass_with_global_model(self, batch: dict):
        self.to_device(batch)
        model_input = {
            'input_ids': batch['concept'],
            'attention_mask': batch['attention_mask'] if 'attention_mask' in batch else None,
            'age_ids': batch['age'] if 'age' in batch else None,
            'segment_ids': batch['segment'] if 'segment' in batch else None,
            'record_rank_ids': batch['record_rank'] if 'record_rank' in batch else None,
            'domain_ids': batch['domain'] if 'domain' in batch else None,
            'target': batch['target'] if 'target' in batch else None,
        }
        if 'plos_target' in batch:
            model_input['plos_target'] = batch['plos_target']
        
        tmp = self.local_model(**model_input)
        attention = tmp.attentions[-1].mean(axis=1).mean(axis=1)
        attention = attention[:, 2:] # [batch size, 2046]

        drop_keys = ['input_ids', 'attention_mask', 'age_ids', 'segment_ids', 'record_rank_ids']
        dropped_model_input = {
            k: [] for k in drop_keys
        }
        for n in range(attention.shape[0]):
            input_ids = model_input['input_ids'][n, 2:] # [2046]
            nonzero_idx = model_input['attention_mask'][n, 2:].type(torch.BoolTensor).to(self.device)
            if sum(nonzero_idx) < 64: 
                for k in drop_keys:
                    dropped_model_input[k].append(model_input[k][n][2:])
                continue
            q8 = torch.quantile(attention[n][nonzero_idx], 0.8)
            nonzero_idx_clone = nonzero_idx.clone()
            nonzero_idx[nonzero_idx_clone] = attention[n][nonzero_idx_clone] >= q8
            for k in drop_keys:
                dropped_model_input[k].append(
                    torch.cat([
                        model_input[k][n][2:][nonzero_idx], 
                        torch.zeros(len(input_ids)-sum(nonzero_idx)).to(self.device)
                    ]).type(torch.LongTensor).to(self.device)
                )
        dropped_model_input = {
            k: torch.cat([model_input[k][:, :2], torch.stack(v)], dim=1) for k, v in dropped_model_input.items()
        }
        dropped_model_input['target'] = model_input['target']

        return dropped_model_input


    def validate(self, dl, partition=True):
        with torch.no_grad():
            self.global_model.eval()
            val_loss = 0
            val_logits = []
            val_labels = []
            n = 0
            for batch in tqdm(dl):
                dropped_input = self.forward_pass_with_global_model(batch)
                outputs = self.global_model(**dropped_input)
                val_loss += outputs.loss.item()
                val_logits.append(self.softmax(outputs.logits)[:, 1])
                val_labels.append(batch['target'])
                if partition and n > len(dl) / 5:
                    break
                n += 1
            
            self.val_loss = val_loss / (n+1)
            self.val_logits = torch.cat(val_logits).cpu().detach().numpy()
            self.val_labels = torch.cat(val_labels).cpu().detach().numpy()

            fpr, tpr, _ = roc_curve(self.val_labels, self.val_logits)
            self.roc_auc = auc(fpr, tpr)
            
    
    def save_model(self):
        torch.save({
            'epoch': self.epoch,
            'model': self.global_model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, self.args.savepth)
        
            
class EthosTrainer(EHRTrainer):
    def __init__(self, 
        model: torch.nn.Module,
        train_dataset: Dataset = None,
        valid_dataset: Dataset = None,
        args: dict = {},
    ):
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=0.01,
            eps=1e-8,
        )
        self.args = args

    def train(self, gpu=None, ngpus_per_node=None, args=None):
        if self.args.multi_gpu:
            self.args = args
            self.device = f'cuda:{gpu}'
            ngpus_per_node = torch.cuda.device_count()    
            print("Use GPU: {} for training".format(gpu))
        
            self.args.rank = self.args.rank * ngpus_per_node + gpu    
            dist.init_process_group(backend=self.args.dist_backend, init_method=self.args.dist_url,
                                    world_size=self.args.world_size, rank=self.args.rank)
            
            torch.cuda.set_device(gpu)
            self.model.to(self.device)
            self.args.bs = int(self.args.bs / ngpus_per_node)
            # self.args.num_workers = int(self.args.num_workers / ngpus_per_node)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[gpu], find_unused_parameters=True)
        

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.args.logpth)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        first_iter = 0
        if os.path.exists(self.args.savepth):
            print('Saved file exists.')
            checkpoint = torch.load(self.args.savepth, map_location=self.device)
            first_iter = checkpoint["epoch"] + 1
            self.model.load_state_dict(checkpoint["model"], strict=False)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            
        if self.args.multi_gpu:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
            self.train_dl = DataLoader(self.train_dataset, batch_size=self.args.bs, 
                                    shuffle=(train_sampler is None), num_workers=self.args.num_workers, 
                                    sampler=train_sampler, pin_memory=True)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_dataset, shuffle=True)
            self.valid_dl = DataLoader(self.valid_dataset, batch_size=self.args.bs, 
                                    shuffle=(valid_sampler is None), num_workers=self.args.num_workers, 
                                    sampler=valid_sampler, pin_memory=True)
        else:
            self.train_dl = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True)
            self.valid_dl = DataLoader(self.valid_dataset, batch_size=self.args.bs, shuffle=True)

        best_loss = 999999
        prev_time = time.time()
        scaler = torch.amp.GradScaler()
        
        eval_interval = 8000
        patience = 0
        for epoch in range(first_iter, self.args.max_epoch):
            self.epoch = epoch
            self.model.train()
            epoch_loss = 0

            for i, (X, Y) in enumerate(self.train_dl):
                if i == eval_interval: break

                # Train step
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    _, loss = self.forward_pass(X, Y['concept'])

                scaler.scale(loss).backward()
                if (i+1) % 40 == 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item()

                # Determine approximate time left
                batches_done = epoch * eval_interval + i + 1
                batches_left = self.args.max_epoch * eval_interval - batches_done
                time_left = datetime.timedelta(
                    seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [Train batch loss: %f] ETA: %s"
                    % (epoch+1, self.args.max_epoch, i+1, eval_interval, loss.item(), time_left)
                )
                

            # Validate (returns None if no validation set is provided)
            # RuntimeError of dataloader occurs when using multi-gpu due to "break" below.
            # You can ignore the RuntimeError.
            eval_size = 1000
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for i, (X, Y) in enumerate(self.valid_dl):
                    _, loss = self.forward_pass(X, Y['concept'])
                    val_loss += loss.item()
                    if i == eval_size: break
            
            self.val_loss = val_loss / eval_size

            valid_flag = False
            if not self.args.multi_gpu:
                valid_flag = True
            elif self.args.rank == 0:
                valid_flag = True
                    
            if valid_flag:
                # Print epoch info
                self.logger.info('Epoch [{}/{}], Train Loss: {:.4f} Val Loss: {:.4f}'.format(
                    epoch+1, self.args.max_epoch, epoch_loss / eval_interval, self.val_loss))

                if self.val_loss < best_loss:
                    self.logger.info('Save best model...')
                    self.save_model()
                    best_loss = self.val_loss
                    patience = 0
                else:
                    patience += 1
                    if patience >= 10:
                        break


    def forward_pass(self, batch, target):
        self.to_device(batch)
        target = target.to(self.device)
        model_input = {
            'input_ids': batch['concept'],
            'age_ids': batch['age'] if 'age' in batch else None,
            'segment_ids': batch['segment'] if 'segment' in batch else None,
            'record_rank_ids': batch['record_rank'] if 'record_rank' in batch else None,
            'domain_ids': batch['domain'] if 'domain' in batch else None,
            'target': target,
        }
        if 'plos_target' in batch:
            model_input['plos_target'] = batch['plos_target']
        return self.model(**model_input)


    def validation(self):
        pass

        