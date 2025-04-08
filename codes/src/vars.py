import sys
sys.path.append('..')
from src.utils import DotDict


tokenizer_config = DotDict({
    'sep_tokens': True, # should we add [SEP] tokens?
    'cls_token': True, # should we add a [CLS] token?
    'padding': True, # should we pad the sequences?
    'truncation': 2048}) # how long should the longest sequence be


outcome_prediction_point = {
    'mortality': 'visit_start_datetime',
    'LLOS': 'visit_start_datetime',
    'readmission': 'visit_end_datetime',
}