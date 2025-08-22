import numpy as np
import os
from tqdm import tqdm
import time
import datetime
import logging
import traceback
import copy
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torch.optim import AdamW
import torch.distributed as dist
from transformers import BertConfig

import sys
from pathlib import Path
abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)

from src.stat_function import (
    auc_ci, prc_ci, calculate_confidence_interval,
    calculate_sensitivity, calculate_specificity, 
    calculate_precision, calculate_f1_score
)
from src.utils import r3, picklesave, pickleload
from src.baseline_models import EHRBertForSequenceClassification
abspath = str(Path(__file__).resolve().parent.parent.parent)
spath = os.path.join(abspath, 'usedata/{hospital}/')
tpath = os.path.join(abspath, 'usedata/{hospital}/tokens/Finetuning')


class EHRTrainer:
    def __init__(self, 
        model: torch.nn.Module,
        train_dataset: Dataset = None,
        valid_dataset: Dataset = None,
        args: dict = {},
    ):
        self.model = model
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
        import sys
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

        train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
        self.train_dl = DataLoader(self.train_dataset, batch_size=self.args.bs, 
                                shuffle=(train_sampler is None), num_workers=self.args.num_workers, 
                                sampler=train_sampler, pin_memory=True)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_dataset, shuffle=False)
        self.valid_dl = DataLoader(self.valid_dataset, batch_size=self.args.bs, 
                                shuffle=(valid_sampler is None), num_workers=self.args.num_workers, 
                                sampler=valid_sampler, pin_memory=True)

        best_loss = 999999
        prev_time = time.time()
        scaler = torch.amp.GradScaler()
        save_flag = False
        patience = 0
        try:
            for epoch in range(self.args.first_iter, self.args.max_epoch):
                self.epoch = epoch
                self.model.train()
                epoch_loss = 0
                stop_flag = torch.tensor([0], device=self.device)


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
                    num_batches = 0
                    for n, batch in enumerate(self.valid_dl):
                        outputs = self.forward_pass(batch)
                        val_loss += outputs.loss.item()
                        num_batches += 1
                        sys.stdout.write(
                            "\rValidation: %d/%d"
                            % (n+1, len(self.valid_dl))
                        )

                val_loss_tensor = torch.tensor(val_loss, device=self.device) 
                num_batches_tensor = torch.tensor(num_batches, device=self.device) 

                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
                self.val_loss = (val_loss_tensor / num_batches_tensor).item()

                valid_flag = False
                if self.args.rank == 0:
                    valid_flag = True
                        
                if valid_flag:
                    # Print epoch info
                    self.logger.info('Epoch [{}/{}], Train Loss: {:.4f} Val Loss: {:.4f}'.format(
                        epoch+1, self.args.max_epoch, epoch_loss / len(self.train_dl), self.val_loss))

                    if self.val_loss < best_loss:
                        self.logger.info('Save best model...')
                        self.save_model()
                        best_loss = self.val_loss
                        patience = 0
                    
                    else:
                        patience += 1
                        if patience >= 5: 
                            self.logger.info('Early stopping activated.')
                            stop_flag[0] = 1
                        
                dist.broadcast(stop_flag, src=0)
                if stop_flag.item() == 1: break
        
        except KeyboardInterrupt:
            import sys
            sys.exit('KeyboardInterrupt')

        except:
            logging.error(traceback.format_exc())
        


    def forward_pass(self, batch: dict, inference=False, output_attentions=False):
        self.to_device(batch)
        model_input = {
            'input_ids': batch['concept'],
            'attention_mask': batch['attention_mask'] if 'attention_mask' in batch else None,
            'age_ids': batch['age'] if 'age' in batch else None,
            'segment_ids': batch['segment'] if 'segment' in batch else None,
            'record_rank_ids': batch['record_rank'] if 'record_rank' in batch else None,
            'domain_ids': batch['domain'] if 'domain' in batch else None,
            'target': batch['target'] if 'target' in batch else None,
            'output_attentions': output_attentions,
        }
        if 'plos_target' in batch:
            model_input['plos_target'] = batch['plos_target']
        output = self.model(**model_input, inference=inference) if inference else self.model(**model_input)
        return output
        

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
        dataset: torch.utils.data.Dataset = None,
        ex_dataset: torch.utils.data.Dataset = None,
        logit_path: str = '',
        vocab=None,
        ex_vocab=None,
        selected_representation=None,
        ex_selected_representation=None,
        train_size=0.6,
        test_size=0.2,
        args: dict = {},
    ):
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        self.dataset = dataset
        self.ex_dataset = ex_dataset
        self.logit_path = logit_path
        self.vocab = vocab
        self.ex_vocab = ex_vocab
        self.selected_representation = selected_representation
        self.ex_selected_representation = ex_selected_representation
        self.train_size = train_size
        self.test_size = test_size
        self.softmax = torch.nn.Softmax(dim=1)
        self.args = args
            

    def process(self, model, args=None):
        import sys
        try:
            self.model = model.to(self.device)
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=0.01,
                eps=1e-8,
            )
            self.args = args
            self.device = f'cuda:{self.args.device}'
            self.model.to(self.device)
                
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(self.args.logpth)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

            initial_weight = copy.deepcopy(self.model.state_dict())

            # for seed in (100, ):
            for seed in (100, 200, 300):
                if os.path.exists(self.logit_path + f'_{seed}.pkl'): 
                    if self.ex_dataset is None:
                        continue
                    else:
                        if os.path.exists(self.logit_path + f'_ex_{seed}.pkl'): 
                            continue
                self.logger.info(f"\n\n***Seed: {seed}***\n\n")
                self.model.load_state_dict(initial_weight)
                datasets = {}
                generator = torch.Generator().manual_seed(seed)
                train_size = int(self.train_size * len(self.dataset))
                test_size = int(self.test_size * len(self.dataset))
                datasets['train'], valid_dataset = random_split(
                    self.dataset, 
                    [train_size, len(self.dataset) - train_size], 
                    generator=generator)
                datasets['valid'], datasets['test'] = random_split(
                    valid_dataset, 
                    [len(valid_dataset) - test_size, test_size], 
                    generator=generator)

                self.dls = {}
                for k, v in datasets.items():
                    _v = copy.deepcopy(v)
                    is_train = k == 'train'
                    if not is_train:
                        _v.dataset.dataset.limit_aug = True
                    dl = DataLoader(_v, batch_size=self.args.bs, shuffle=is_train)
                    self.dls[k] = dl
                
                if self.ex_dataset is not None:
                    self.ex_dataset.limit_aug = True
                    self.ex_dl = DataLoader(self.ex_dataset, batch_size=self.args.bs, shuffle=False)
                self.train(seed)

        except KeyboardInterrupt:
            import sys
            sys.exit('KeyboardInterrupt')

        except:
            logging.error(traceback.format_exc())

    def train(self, seed):
        best_score = 0
        patience = 0
        prev_time = time.time()
        import sys
        if not os.path.exists(self.args.savepth + f'_{seed}.tar'):
            for epoch in range(self.args.max_epoch):
                self.optimizer.zero_grad()
                stop_flag = torch.tensor([0], device=self.device)
                self.epoch = epoch
                epoch_loss = 0
                for i, batch in enumerate(self.dls['train']):
                    self.model.train()
                    # Train step
                    outputs = self.forward_pass(batch)
                    loss = outputs.loss
                    loss.backward()

                    if (i+1) % 4 == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    epoch_loss += loss.item()

                    # Determine approximate time left
                    batches_done = epoch * len(self.dls['train']) + i + 1
                    batches_left = self.args.max_epoch * len(self.dls['train']) - batches_done
                    time_left = datetime.timedelta(
                        seconds=batches_left * (time.time() - prev_time))
                    prev_time = time.time()

                    sys.stdout.write(
                        "\r[Epoch %d/%d] [Batch %d/%d] [Train batch loss: %f] ETA: %s"
                        % (epoch+1, self.args.max_epoch, i+1, len(self.dls['train']), loss.item(), time_left)
                    )
                self.validate(self.dls['valid'])
                    
                # Print epoch info
                self.logger.info('Epoch [{}/{}], Batch [{}/{}], Train Loss: {:.4f} Val Loss: {:.4f} Val AUROC: {:.4f}'.format(
                    epoch+1, self.args.max_epoch, i+1, len(self.dls['train']), epoch_loss / len(self.dls['train']), self.val_loss, self.roc_auc))

                if self.roc_auc > best_score + 0.001:
                    self.logger.info('Save best model...')
                    # self.save_model()
                    best_weight = self.model.state_dict()
                    best_score = self.roc_auc
                    patience = 0
                else:
                    patience += 1
                    if patience >= 5: 
                        self.logger.info('Early stopping activated.')
                        stop_flag[0] = 1
                        
                if stop_flag.item() == 1 or epoch + 1 == self.args.max_epoch: 
                    self.model.load_state_dict(best_weight)
                    self.save_model(self.args.savepth + f'_{seed}.tar')
                    break

        # Load test datasets
        checkpoint = torch.load(self.args.savepth + f'_{seed}.tar', map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        with torch.no_grad():
            if not os.path.exists(self.logit_path + f'_{seed}.pkl'):
                self.validate(self.dls['test'])
                picklesave((self.val_logits, self.val_labels), self.logit_path + f'_{seed}.pkl')
            
            
            if self.ex_dataset is not None:
                if not os.path.exists(self.logit_path + f'_ex_{seed}.pkl'):
                    if self.args.rep_type != 'none':
                        self.model.bert.embeddings.concept_embeddings.weight.data = torch.Tensor(self.ex_selected_representation).to(self.device)
                    self.validate(self.ex_dl)
                    picklesave((self.val_logits, self.val_labels), self.logit_path + f'_ex_{seed}.pkl')
                    if self.args.rep_type != 'none':
                        self.model.bert.embeddings.concept_embeddings.weight.data = torch.Tensor(self.selected_representation).to(self.device)
                        for n, p in self.model.bert.embeddings.named_parameters():
                            print(n, p.requires_grad)

    def validate(self, dl):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0
            val_logits = []
            val_labels = []
            
            for n, batch in enumerate(dl):
                outputs = self.forward_pass(batch)
                val_loss += outputs.loss.item()
                if outputs.logits.shape[1] == 2:
                    val_logits.append(self.softmax(outputs.logits)[:, 1])
                else:
                    val_logits.append(outputs.logits)
                val_labels.append(batch['target'])
                sys.stdout.write(
                    "\rValidation: %d/%d"
                    % (n+1, len(dl))
                )
            
            self.val_loss = val_loss / (n+1)
            self.val_logits = torch.cat(val_logits)
            self.val_labels = torch.cat(val_labels)

            self.val_logits = self.val_logits.cpu().detach().numpy()
            self.val_labels = self.val_labels.cpu().detach().numpy()
            if outputs.logits.shape[1] == 2:
                fpr, tpr, _ = roc_curve(self.val_labels, self.val_logits)
                self.roc_auc = auc(fpr, tpr)
            else:
                all_roc_auc = []
                for _n in range(outputs.logits.shape[1]):
                    fpr, tpr, _ = roc_curve(self.val_labels[:, _n], self.val_logits[:, _n])
                    all_roc_auc.append(auc(fpr, tpr))
                self.roc_auc = np.mean(all_roc_auc)

            
    def save_model(self, savepth):
        torch.save({
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, savepth)

            
    def extract_attention_score(self, model, args=None):
        self.model = model.to(self.device)
        self.args = args
        self.device = f'cuda:{self.args.device}'
        self.model.to(self.device)

        # for seed in (100, ):
        for seed in (100, 200, 300):
            load_model = torch.load(self.args.savepth + f'_{seed}.tar', map_location=self.device)['model']
            self.model.load_state_dict(load_model)

            datasets = {}
            generator = torch.Generator().manual_seed(seed)
            train_size = int(self.train_size * len(self.dataset))
            test_size = int(self.test_size * len(self.dataset))
            datasets['train'], valid_dataset = random_split(
                self.dataset, 
                [train_size, len(self.dataset) - train_size], 
                generator=generator)
            datasets['valid'], datasets['test'] = random_split(
                valid_dataset, 
                [len(valid_dataset) - test_size, test_size], 
                generator=generator)

            self.dls = {}
            for k, v in datasets.items():
                _v = copy.deepcopy(v)
                is_train = k == 'train'
                if not is_train:
                    _v.dataset.dataset.limit_aug = True
                dl = DataLoader(_v, batch_size=int(self.args.bs / 4), shuffle=is_train)
                self.dls[k] = dl
            self.in_dl = self.dls['test']

            if self.ex_dataset is not None:
                self.ex_dataset.limit_aug = True
                self.ex_dl = DataLoader(self.ex_dataset, batch_size=int(self.args.bs / 4), shuffle=False)
            
            self.model.eval()
            with torch.no_grad():
                seqs = []
                last_cls_atts = []
                if self.args.rep_type != 'none':
                    self.model.bert.embeddings.concept_embeddings.weight.data = torch.Tensor(self.selected_representation).to(self.device)
                prev_time = time.time()
                if not os.path.exists(self.logit_path.replace('logits', 'atts') + f'_last_cls_{seed}.pkl'):
                    for n, batch in enumerate(self.in_dl):
                        outputs = self.forward_pass(batch, output_attentions=True)
                        atts = [o.cpu().detach() for o in outputs.attentions]
                        seqs.append(batch['concept'].cpu().detach())
                        last_cls_atts.append(atts[-1].mean(dim=(1,))[:, 0])

                        # Determine approximate time left
                        batches_done = n + 1
                        batches_left = len(self.in_dl) - batches_done
                        time_left = datetime.timedelta(
                            seconds=batches_left * (time.time() - prev_time))
                        prev_time = time.time()
                        sys.stdout.write(
                            "\r[Batch %d/%d] ETA: %s"
                            % (n+1, len(self.in_dl), time_left)
                        )
                        
                    picklesave(torch.cat(seqs), self.logit_path.replace('logits', 'atts') + f'_seq_{seed}.pkl')
                    picklesave(torch.cat(last_cls_atts), self.logit_path.replace('logits', 'atts') + f'_last_cls_{seed}.pkl')
                
                seqs = []
                last_cls_atts = []
                prev_time = time.time()
                if self.args.rep_type != 'none':
                    self.model.bert.embeddings.concept_embeddings.weight.data = torch.Tensor(self.ex_selected_representation).to(self.device)
                if not os.path.exists(self.logit_path.replace('logits', 'atts') + f'_ex_last_cls_{seed}.pkl'):
                    for n, batch in enumerate(self.ex_dl):
                        outputs = self.forward_pass(batch, output_attentions=True)
                        atts = [o.cpu().detach() for o in outputs.attentions]
                        seqs.append(batch['concept'].cpu().detach())
                        last_cls_atts.append(atts[-1].mean(dim=(1,))[:, 0])

                        # Determine approximate time left
                        batches_done = n + 1
                        batches_left = len(self.ex_dl) - batches_done
                        time_left = datetime.timedelta(
                            seconds=batches_left * (time.time() - prev_time))
                        prev_time = time.time()
                        sys.stdout.write(
                            "\r[Batch %d/%d] ETA: %s"
                            % (n+1, len(self.in_dl), time_left)
                        )
                        
                    picklesave(torch.cat(seqs), self.logit_path.replace('logits', 'atts') + f'_ex_seq_{seed}.pkl')
                    picklesave(torch.cat(last_cls_atts), self.logit_path.replace('logits', 'atts') + f'_ex_last_cls_{seed}.pkl')
                
                if self.args.rep_type != 'none':
                    self.model.bert.embeddings.concept_embeddings.weight.data = torch.Tensor(self.selected_representation).to(self.device)
                

class MixEHRTrainer(EHRTrainer):
    def __init__(self, 
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        logit_path: str = '',
        args: dict = {},
    ):
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.dataset = dataset
        self.logit_path = logit_path
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=0.01,
            eps=1e-8,
        )
        self.softmax = torch.nn.Softmax(dim=1)
        self.args = args
            

    def train(self, gpu=None, ngpus_per_node=None, args=None):
        import sys
        try:
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

            initial_weight = copy.deepcopy(self.model.state_dict())

            for seed in (100, 200, 300, 400, 500):
                dist.barrier()
                if os.path.exists(self.logit_path + f'_{seed}.pkl'): 
                    continue
                if dist.get_rank() == 0: 
                    self.logger.info(f"\n\n***Seed: {seed}***\n\n")
                
                self.model.load_state_dict(initial_weight)
                datasets = {}
                generator = torch.Generator().manual_seed(seed)
                train_size = int(0.6 * len(self.dataset))
                test_size = int(0.2 * len(self.dataset))
                datasets['train'], valid_dataset = random_split(
                    self.dataset, 
                    [train_size, len(self.dataset) - train_size], 
                    generator=generator)
                datasets['valid'], datasets['test'] = random_split(
                    valid_dataset, 
                    [test_size, len(valid_dataset) - test_size], 
                    generator=generator)

                dls = {}
                for k, v in datasets.items():
                    shuffle = k == 'train'
                    sampler = torch.utils.data.distributed.DistributedSampler(v, shuffle=shuffle)
                    if k != 'test':
                        dl = DataLoader(v, batch_size=self.args.bs,
                                        shuffle=(sampler is None), num_workers=self.args.num_workers, 
                                        sampler=sampler, pin_memory=True)
                    else:
                        dl = DataLoader(v, batch_size=self.args.bs, shuffle=False)
                    dls[k] = dl

                best_score = 0
                patience = 0
                prev_time = time.time()

                import sys
                for epoch in range(self.args.max_epoch):
                    self.optimizer.zero_grad()
                    stop_flag = torch.tensor([0], device=self.device)
                    self.epoch = epoch
                    epoch_loss = 0
                    for i, batch in enumerate(dls['train']):
                        self.model.train()
                        # Train step
                        outputs = self.forward_pass(batch)
                        loss = outputs.loss
                        loss.backward()

                        if (i+1) % 4 == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                        epoch_loss += loss.item()

                        # Determine approximate time left
                        batches_done = epoch * len(dls['train']) + i + 1
                        batches_left = self.args.max_epoch * len(dls['train']) - batches_done
                        time_left = datetime.timedelta(
                            seconds=batches_left * (time.time() - prev_time))
                        prev_time = time.time()

                        sys.stdout.write(
                            "\r[Epoch %d/%d] [Batch %d/%d] [Train batch loss: %f] ETA: %s"
                            % (epoch+1, self.args.max_epoch, i+1, len(dls['train']), loss.item(), time_left)
                        )

                    self.model.eval()
                    with torch.no_grad():
                        self.validate(dls['valid'])
                        
                    # Print epoch info
                    if dist.get_rank() == 0:
                        self.logger.info('Epoch [{}/{}], Batch [{}/{}], Train Loss: {:.4f} Val Loss: {:.4f} Val AUROC: {:.4f}'.format(
                            epoch+1, self.args.max_epoch, i+1, len(dls['train']), epoch_loss / len(dls['train']), self.val_loss, self.roc_auc))

                        if self.roc_auc > best_score + 0.001:
                            self.logger.info('Save best model...')
                            # self.save_model()
                            best_weight = self.model.state_dict()
                            best_score = self.roc_auc
                            patience = 0
                        else:
                            patience += 1
                            if patience >= 5: 
                                self.logger.info('Early stopping activated.')
                                stop_flag[0] = 1
                            
                    dist.broadcast(stop_flag, src=0)
                    if stop_flag.item() == 1 or epoch + 1 == self.args.max_epoch: 
                        if dist.get_rank() == 0:
                            self.model.load_state_dict(best_weight)
                            self.save_model()
                        break

                # Load test datasets
                dist.barrier()
                if dist.get_rank() == 0:
                    checkpoint = torch.load(self.args.savepth, map_location=self.device)
                    self.model.load_state_dict(checkpoint["model"])
                    with torch.no_grad():
                        self.validate(dls['test'], True)
                    picklesave((self.val_logits, self.val_labels), self.logit_path + f'_{seed}.pkl')

        except KeyboardInterrupt:
            import sys
            sys.exit('KeyboardInterrupt')

        except:
            logging.error(traceback.format_exc())


    def validate(self, dl, test=False):
        with torch.no_grad():
            self.model.eval()
            val_loss = 0
            val_logits = []
            val_labels = []
            
            for n, batch in enumerate(dl):
                outputs = self.forward_pass(batch, True)
                val_loss += outputs.loss.item()
                val_logits.append(self.softmax(outputs.logits)[:, 1])
                val_labels.append(batch['target'])
                sys.stdout.write(
                    "\rValidation: %d/%d"
                    % (n+1, len(dl))
                )
            
            self.val_loss = val_loss / (n+1)
            self.val_logits = torch.cat(val_logits)
            self.val_labels = torch.cat(val_labels)

            if test:
                self.val_logits = self.val_logits.cpu().detach().numpy()
                self.val_labels = self.val_labels.cpu().detach().numpy()
                fpr, tpr, _ = roc_curve(self.val_labels, self.val_logits)
                self.roc_auc = auc(fpr, tpr)
            else:
                # --- All gather across GPUs ---
                gathered_logits = [torch.zeros_like(self.val_logits) for _ in range(dist.get_world_size())]
                gathered_labels = [torch.zeros_like(self.val_labels) for _ in range(dist.get_world_size())]

                dist.all_gather(gathered_logits, self.val_logits)
                dist.all_gather(gathered_labels, self.val_labels)

                # --- Combine all gathered tensors ---
                self.val_logits = torch.cat(gathered_logits).cpu().detach().numpy()
                self.val_labels = torch.cat(gathered_labels).cpu().detach().numpy()
                dist.barrier()

                # --- Calculate AUROC only on one process (e.g., rank 0) ---
                if dist.get_rank() == 0:
                    fpr, tpr, _ = roc_curve(self.val_labels, self.val_logits)
                    self.roc_auc = auc(fpr, tpr)
                else:
                    self.roc_auc = 0.0  # or use dist broadcast if needed
            