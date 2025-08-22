import torch.nn as nn
import torch


class EhrEmbeddings(nn.Module):
    """
        EHR Embeddings

        Forward inputs:
            input_ids: torch.LongTensor             - (batch_size, sequence_length)
            token_type_ids: torch.LongTensor        - (batch_size, sequence_length)
            position_ids: dict(str, torch.Tensor)   - (batch_size, sequence_length)

        Config:
            vocab_size: int                         - size of the vocabulary
            hidden_size: int                        - size of the hidden layer
            type_vocab_size: int                    - size of max segments
            layer_norm_eps: float                   - epsilon for layer normalization
            hidden_dropout_prob: float              - dropout probability
            linear: bool                            - whether to linearly scale embeddings (a: concept, b: age, c: abspos, d: segment)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
       
        self.concept_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.age_embeddings = nn.Embedding(config.max_age, config.hidden_size)
        self.visit_rank_embeddings = nn.Embedding(config.max_visit_rank, config.hidden_size)
        self.record_rank_embeddings = nn.Embedding(config.max_record_rank, config.hidden_size)
        self.domain_embeddings = nn.Embedding(config.max_domain, config.hidden_size) # condition, measurement, drug, procedure + alpha

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.config.to_dict().get('linear', False):
            self.a = nn.Parameter(torch.ones(1))
            self.b = nn.Parameter(torch.zeros(1))
            self.c = nn.Parameter(torch.zeros(1))
            self.d = nn.Parameter(torch.zeros(1))
            self.e = nn.Parameter(torch.zeros(1))
        else:
            self.a = self.b = self.c = self.d = self.e = 1

    def forward(
        self,
        input_ids: torch.LongTensor,
        age_ids: torch.LongTensor = None,
        segment_ids: torch.LongTensor = None,
        record_rank_ids: torch.LongTensor = None,
        domain_ids: torch.LongTensor = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs
    ):
        if inputs_embeds is not None:
            return inputs_embeds

        embeddings = self.a * self.concept_embeddings(input_ids)
        
        if age_ids is not None:
            ages_embedded = self.age_embeddings(age_ids)
            embeddings += self.b * ages_embedded

        if segment_ids is not None:
            visit_rank_embedded = self.visit_rank_embeddings(segment_ids)
            embeddings += self.c * visit_rank_embedded

        if record_rank_ids is not None:
            record_rank_embedded = self.record_rank_embeddings(record_rank_ids)
            embeddings += self.d * record_rank_embedded

        if domain_ids is not None:
            types_embeddings = self.domain_embeddings(domain_ids)
            embeddings += self.e * types_embeddings
            
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

