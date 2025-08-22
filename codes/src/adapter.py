import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import (
    BertLayer, apply_chunking_to_forward, BertEncoder
    )
from typing import List, Optional, Tuple, Union

class BottleneckAdapter(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        bottleneck_size: int
    ):
        super().__init__()
        self.activation = nn.GELU()
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
    
    def forward(self, x, residual=True):
        x_original = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        if residual:
            return x_original + x
        else:
            return x


class BertLayerWithAdapter(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        
        self.adapter = BottleneckAdapter(
            hidden_size=config.hidden_size, 
            bottleneck_size=int(config.hidden_size / 2)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        outputs = super().forward(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions
        )
        
        layer_output = outputs[0]                    
        layer_output = self.adapter(layer_output)    

        return (layer_output,) + outputs[1:]


class AdapterEmbeddedBertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        module_list = []
        for _ in range(config.num_hidden_layers):
            module_list.append(BertLayerWithAdapter(config))

        self.layer = nn.ModuleList(module_list)


class AdapterFusion(nn.Module):
    def __init__(self, hidden_size, num_adapters, query_dim=None, ):
        super().__init__()

        self.hidden_size = hidden_size          # e.g., 768 (BERT base)
        self.num_adapters = num_adapters        # e.g., 3 if using 3 adapters
        self.query_dim = query_dim or hidden_size

        # Q, K, V projection for attention
        self.query_proj = nn.Linear(hidden_size, self.query_dim)       # (hidden_size → query_dim)
        self.key_proj   = nn.Linear(hidden_size, self.query_dim)       # (hidden_size → query_dim)
        self.value_proj = nn.Linear(hidden_size, hidden_size)          # (hidden_size → hidden_size)

        # Optional output projection after fusion
        self.output_proj = nn.Linear(hidden_size, hidden_size)         # (hidden_size → hidden_size)
