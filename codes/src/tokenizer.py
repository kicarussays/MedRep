import torch
import typer
import pickle as pkl
from os.path import join, split
from tqdm import tqdm
from transformers import BatchEncoding


class EHRTokenizer():
    def __init__(self, vocabulary=None, config=None):
        self.frozen = False
        self.config = config
        if isinstance(vocabulary, type(None)):
            self.vocabulary = {
                '[PAD]':0,
                '[MASK]':1,
                '[UNK]':2,
                '[CLS]':3,
                '[SEP]':4,
            }
            
        else:
            self.vocabulary = vocabulary

    def __call__(self, features):
        return self.batch_encode(features)

    def encode(self, concepts):
        if not self.frozen:
            for concept in concepts:
                if concept not in self.vocabulary:
                    self.vocabulary[concept] = len(self.vocabulary)
                
        return [self.vocabulary.get(concept, self.vocabulary['[UNK]']) for concept in concepts]
   
    @staticmethod
    def truncate(patient: dict, max_len: int):

        for key, value in patient.items():
            patient[key] = value[:max_len]    # Keep background sentence + newest information

        return patient

    def batch_encode(self, features: dict):
        data = {key: [] for key in features}
        data['attention_mask'] = []

        for patient in tqdm(self._patient_iterator(features), desc="Encoding patients"):
            patient = self.insert_special_tokens(patient)                   # Insert SEP and CLS tokens

            if self.config.truncation and (len(patient['concept']) >  self.config.truncation):
                patient = self.truncate(patient, max_len= self.config.truncation)        # Truncate patient to max_len
            
            # Created after truncation for efficiency
            patient['attention_mask'] = [1] * len(patient['concept'])       # Initialize attention mask
            patient['concept'] = self.encode(patient['concept'])            # Encode concepts

            for key, value in patient.items():
                data[key].append(value)

        if self.config.padding:
            longest_seq = self.config.truncation            # Find longest sequence
            data = self.pad(data, max_len=longest_seq)                      # Pad sequences to max_len
        return BatchEncoding(data, tensor_type='pt' if self.config.padding else None)
    
    def insert_special_tokens(self, patient: dict):
        if self.config['sep_tokens']:
            if 'segment' not in patient:
                raise Exception('Cannot insert [SEP] tokens without segment information')
            patient = self.insert_sep_tokens(patient)

        if self.config.cls_token:
            patient = self.insert_cls_token(patient)
        return patient
    
    def insert_sep_tokens(self, patient: dict):
        padded_segment = patient['segment'] + [None]                # Add None to last entry to avoid index out of range

        for key, values in patient.items():
            new_seq = []
            for i, val in enumerate(values):
                new_seq.append(val)

                if padded_segment[i] != padded_segment[i+1]:
                    token = '[SEP]' if key == 'concept' else val
                    new_seq.append(token)

            patient[key] = new_seq

        return patient
    
    def insert_cls_token(self, patient: dict):
        for key, values in patient.items():
            if key == 'concept':
                token = '[CLS]'
            elif key == 'age':
                token = values[0]
            else:
                token = 0          # Determine token value (CLS for concepts, 0 for rest)
            patient[key] = [token] + values
        return patient
        
    def _patient_iterator(self, features: dict):
        for i in range(len(features['concept'])):
            yield {key: values[i] for key, values in features.items()}

    def pad(self, features: dict,  max_len: int):
        padded_data = {key: [] for key in features}
        for patient in self._patient_iterator(features):
            difference = max_len - len(patient['concept'])

            for key, values in patient.items():
                token = self.vocabulary['[PAD]'] if key == 'concept' else 0
                padded_data[key].append(values + [token] * difference)

        return padded_data

    def freeze_vocabulary(self):
        self.frozen = True
        
    def save_vocab(self, dest):
        print(f"Writing vocab to {dest}")
        torch.save(self.vocabulary, dest)

