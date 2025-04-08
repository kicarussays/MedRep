from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np
from bisect import bisect
from typing import Sequence, Callable


class BaseDataset(Dataset):
    def __init__(self, features: dict, **kwargs):
        self.features = features
        self.kwargs = kwargs
        # self.max_segments = self.get_max_segments()
        self.max_segments = 2048

    def __len__(self):
        return len(self.features['concept'])

    def __getitem__(self, index):
        return {key: values[index] for key, values in self.features.items()}

    def get_max_segments(self):
        if 'segment' not in self.features:
            raise ValueError('No segment data found. Please add segment data to dataset')
        return max([max(segment) for segment in tqdm(self.features['segment'])]) + 1

    def load_vocabulary(self, vocabulary):
        if isinstance(vocabulary, str):
            return torch.load(vocabulary)
        elif isinstance(vocabulary, dict):
            return vocabulary
        else:
            raise TypeError(f'Unsupported vocabulary input {type(vocabulary)}')
        
    def convert_to_long(self, patient):
        """
        Converts all tensors in the patient to longs except abspos
        """
        for k, v in patient.items():
            if isinstance(v, torch.Tensor) and k != 'abspos':
                patient[k] = v.long()
        return patient


class MLM_Dataset(BaseDataset):
    def __init__(self, features: dict, input_ids='concept', target_ids='target', **kwargs):
        super().__init__(features, **kwargs)
        
        self.vocabulary = self.load_vocabulary(self.kwargs.get('vocabulary', 'vocabulary.pt'))
        self.masked_ratio = self.kwargs.get('masked_ratio', 0.3)
        self.input_ids = input_ids
        self.target_ids = target_ids
        if kwargs.get('ignore_special_tokens', True):
            self.n_special_tokens = len([token for token in self.vocabulary if token.startswith('[')])
        else:
            self.n_special_tokens = 0

    def __getitem__(self, index):
        patient = super().__getitem__(index)

        masked_concepts, target = self._mask(patient)
        patient[self.input_ids] = masked_concepts
        patient[self.target_ids] = target
        
        patient = self.convert_to_long(patient)
        return patient
    

    def __len__(self):
        return self.features[self.input_ids].shape[0]
        
   
    def _mask(self, patient: dict):
        concepts = patient[self.input_ids]

        N = len(concepts)

        # Initialize
        masked_concepts = torch.clone(concepts)
        target = torch.ones(N, dtype=torch.long) * -100

        # Apply special token mask and create MLM mask
        eligible_mask = masked_concepts >= self.n_special_tokens
        eligible_concepts = masked_concepts[eligible_mask]  # Ignore special tokens
        rng = torch.rand(len(eligible_concepts))  # Random number for each token
        masked = rng < self.masked_ratio  # Mask tokens with probability masked_ratio

        # Get masked MLM concepts
        selected_concepts = eligible_concepts[masked]  # Select set % of the tokens
        adj_rng = rng[masked].div(self.masked_ratio)  # Fix ratio to 0-100 interval

        # Operation masks
        rng_mask = adj_rng < 0.8  # 80% - Mask token
        rng_replace = (0.8 <= adj_rng) & (
            adj_rng < 0.9
        )  # 10% - replace with random word
        # rng_keep = adj_rng >= 0.9                             # 10% - keep token (Redundant)

        # Apply operations (Mask, replace, keep)
        selected_concepts = torch.where(
            rng_mask, self.vocabulary["[MASK]"], selected_concepts
        )  # Replace with [MASK]
        selected_concepts = torch.where(
            rng_replace,
            torch.randint(
                self.n_special_tokens, len(self.vocabulary), (len(selected_concepts),)
            ),
            selected_concepts,
        )  # Replace with random word
        # selected_concepts = torch.where(rng_keep, selected_concepts, selected_concepts)       # Redundant

        # Update outputs (nonzero for double masking)
        target[eligible_mask.nonzero()[:, 0][masked]] = eligible_concepts[
            masked
        ]  # Set "true" token
        masked_concepts[
            eligible_mask.nonzero()[:, 0][masked]
        ] = selected_concepts  # Sets new concepts

        return masked_concepts, target
    

class BinaryOutcomeDataset(BaseDataset): 
    def __init__(
        self, 
        features: dict, 
        outcomes: torch.tensor, 
        vocabulary: dict, 
        neighbors_mapped, 
        aug_times,
        drop=False,
        **kwargs
    ):
        super().__init__(features, **kwargs)
        self.outcomes = outcomes
        self.vocabulary = vocabulary
        self.drop = drop
        
        if neighbors_mapped is not None:
            self.neighbors_mapped = torch.Tensor(neighbors_mapped).type(torch.LongTensor)
            self.num_augment = self.neighbors_mapped.shape[1]
        else:
            self.neighbors_mapped = neighbors_mapped
        self.aug_times = aug_times

    def __getitem__(self, index):
        patient = super().__getitem__(index)
        patient['target'] = self.outcomes[index]
        patient = self.convert_to_long(patient)
        if self.neighbors_mapped is not None and torch.rand(1) > 1 / self.aug_times:
            patient = self._augment(patient)

        return patient
    
    def _augment(self, patient):
        concepts = patient['concept']
        augmented_concepts = torch.clone(concepts)
        rng = torch.rand(len(augmented_concepts))  
        aug_ratio = 0.8 + 0.15 * torch.rand(1) # Between 0.8 and 0.95

        augmented = rng < aug_ratio 
        row_indices = torch.arange(augmented_concepts[augmented].shape[0])
        col_indices = torch.randint(0, self.num_augment, (augmented_concepts[augmented].shape[0],))
        selected_neighbors = self.neighbors_mapped[augmented_concepts[augmented]][row_indices, col_indices]
        augmented_concepts[augmented] = selected_neighbors
        patient['concept'] = augmented_concepts.type(torch.LongTensor)
        return patient

    # def _drop(self, patient):
    #     concepts = patient['concept']
    #     drop_ratio = 0.8 * torch.rand(1) # Between 0.0 and 0.8
    #     valid_concepts = concepts[concepts != 0][2:]
    #     if len(valid_concepts) >= 50:
    #         use_idx = torch.rand(len(valid_concepts)) > drop_ratio
    #         use_concepts = torch.cat([concepts[:2], valid_concepts[use_idx]])
    #         concepts = torch.cat([use_concepts, torch.zeros(len(concepts) - len(use_concepts))])
    #     patient['concept'] = concepts.type(torch.LongTensor)
    #     return patient


class EthosDataset(BaseDataset):
    def __init__(self, features: dict, timeline_len: int=2048):
        self.features = features
        self.timeline_len: int = timeline_len

    def __len__(self) -> int:
        return len(self.features['concept']) - self.timeline_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        patient_x = super().__getitem__(idx)
        patient_y = super().__getitem__(idx)
        for key, values in self.features.items():
            patient_x[key] = values[idx:idx+self.timeline_len]
        for key, values in self.features.items():
            patient_y[key] = values[idx+1:idx+1+self.timeline_len]
        patient_x = self.convert_to_long(patient_x)
        patient_y = self.convert_to_long(patient_y)
        return patient_x, patient_y
        
    def convert_to_long(self, patient):
        """
        Converts all tensors in the patient to longs except abspos
        """
        _patient = {}
        for k, v in patient.items():
            _patient[k] = v.long()
        return _patient


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.inputs = inputs
        self.masked_ratio = 0.3

    def __len__(self):
        return self.inputs['input_ids'].shape[0]

    def __getitem__(self, idx):
        return {key: self.inputs[key][idx] for key in self.inputs.keys()}