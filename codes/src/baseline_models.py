from typing import Optional, Union, Tuple, List
import torch
import torch.nn as nn
from transformers import (BertForMaskedLM, BertForPreTraining,
                          BertForSequenceClassification)
from transformers.models.bert.modeling_bert import (
    BertForMaskedLM, MaskedLMOutput, CrossEntropyLoss, BertModel,
    BaseModelOutputWithPoolingAndCrossAttentions, MSELoss,
    add_start_docstrings_to_model_forward, add_code_sample_docstrings,
    _CHECKPOINT_FOR_DOC, MaskedLMOutput, _CONFIG_FOR_DOC, BERT_INPUTS_DOCSTRING,
    BertEncoder, BertPooler, _CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION, SequenceClassifierOutput, 
    _SEQ_CLASS_EXPECTED_OUTPUT, _SEQ_CLASS_EXPECTED_LOSS, SequenceClassifierOutput,
    MSELoss, BCEWithLogitsLoss, BertForPreTrainingOutput
    )
from transformers import BertModel, BertPreTrainedModel

import sys
from pathlib import Path
abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)
from src.embeddings import EhrEmbeddings


class BertModelLabelAdded(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = EhrEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()


    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        age_ids: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        record_rank_ids: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if segment_ids is None:
            segment_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if record_rank_ids is None:
            record_rank_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if domain_ids is None:
            domain_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            age_ids=age_ids,
            segment_ids=segment_ids,
            record_rank_ids=record_rank_ids,
            domain_ids=domain_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BehrtForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModelLabelAdded(config, add_pooling_layer=False)
        self.representation = self.bert.embeddings.concept_embeddings
        
        
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'paris'",
        expected_loss=0.88,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        age_ids: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        record_rank_ids: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            age_ids=age_ids,
            segment_ids=segment_ids,
            record_rank_ids=record_rank_ids,
            domain_ids=domain_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        representations = self.representation(input_ids)

        masked_lm_loss = None
        if target is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            loss_mse = MSELoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), target.view(-1))
            smoothing_loss = loss_mse(sequence_output, representations)
            if self.config.smooth:
                masked_lm_loss += 0.1 * smoothing_loss

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MedbertForMaskedLM(BertForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModelLabelAdded(config, add_pooling_layer=True)
        self.representation = self.bert.embeddings.concept_embeddings
        
        
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'paris'",
        expected_loss=0.88,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        age_ids: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        record_rank_ids: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        plos_target: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            age_ids=age_ids,
            segment_ids=segment_ids,
            record_rank_ids=record_rank_ids,
            domain_ids=domain_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[0], outputs[1]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        representations = self.representation(input_ids)

        total_loss = None
        if target is not None and plos_target is not None:
            loss_fct = CrossEntropyLoss()
            loss_mse = MSELoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), target.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), plos_target.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            smoothing_loss = loss_mse(sequence_output, representations)
            if self.config.smooth:
                total_loss += 0.1 * smoothing_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class EHRBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModelLabelAdded(config, add_pooling_layer=True)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        age_ids: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        record_rank_ids: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            age_ids=age_ids,
            segment_ids=segment_ids,
            record_rank_ids=record_rank_ids,
            domain_ids=domain_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if target is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (target.dtype == torch.long or target.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), target.squeeze())
                else:
                    loss = loss_fct(logits, target)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), target.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, target)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RETAIN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.LSTM_a = nn.LSTM(input_size=config.hidden_size,
                              hidden_size=config.hidden_size,
                              num_layers=config.num_hidden_layers)
        self.LSTM_b = nn.LSTM(input_size=config.hidden_size,
                              hidden_size=config.hidden_size,
                              num_layers=config.num_hidden_layers)
        self.W_alpha = nn.Linear(config.hidden_size, 1)
        self.W_beta = nn.Linear(config.hidden_size, config.hidden_size)
        self.b_alpha = nn.Parameter(torch.randn(1))
        self.b_beta = nn.Parameter(torch.randn(config.hidden_size))
        self.W = nn.Linear(config.hidden_size, config.hidden_size)
        self.b = nn.Parameter(torch.randn(config.hidden_size))
        self.softmax = nn.Softmax(dim=0)
        self.tanh = nn.Tanh()
    

    def forward(self, v):
        g = self.LSTM_a(v)[0]
        e = self.W_alpha(g) + self.b_alpha
        alpha = self.softmax(e.view(e.shape[0], -1))

        h = self.LSTM_b(v)[0]
        beta = self.tanh(self.W_beta(h) + self.b_beta)

        c = alpha.view(alpha.shape[0], -1, 1) * beta * v
        c = torch.sum(c, dim=1)

        y = self.W(c) + self.b

        return y


    def attention(self, v):
        g = self.LSTM_a(v)[0]
        e = self.W_alpha(g) + self.b_alpha
        alpha = self.softmax(e.view(e.shape[0], -1))

        return alpha


class RNNcustom(nn.Module):
    def __init__(self, config, rnn):
        super().__init__()
        if rnn == 'lstm':
            self.RNN = nn.LSTM(input_size=config.hidden_size,
                               hidden_size=config.hidden_size,
                               num_layers=config.num_hidden_layers)
        elif rnn == 'gru':
            self.RNN = nn.GRU(input_size=config.hidden_size,
                               hidden_size=config.hidden_size,
                               num_layers=config.num_hidden_layers)
        elif rnn == 'rnn':
            self.RNN = nn.RNN(input_size=config.hidden_size,
                               hidden_size=config.hidden_size,
                               num_layers=config.num_hidden_layers)
    
    def forward(self, x):
        x = self.RNN(x)[0]
        return x[:, -1, :]



class RNNForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, rnn, attention=False):
        super().__init__(config)

        self.rnn = rnn
        self.attention = attention
        self.config = config
        self.embeddings = EhrEmbeddings(config)
        
        if rnn in ('lstm', 'gru'):
            self.RNN = RNNcustom(config, rnn)
            
        
        elif rnn == 'retain':
            self.RNN = RETAIN(config)


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        age_ids: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        record_rank_ids: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        embedding_output = self.embeddings(
            input_ids=input_ids,
            age_ids=age_ids,
            segment_ids=segment_ids,
            record_rank_ids=record_rank_ids,
            domain_ids=domain_ids,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = self.RNN(embedding_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if target is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (target.dtype == torch.long or target.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), target.squeeze())
                else:
                    loss = loss_fct(logits, target)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), target.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, target)

        if self.attention: return self.RNN.attention(embedding_output)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


# Code from official repository of ETHOS 
# https://github.com/ipolharvard/ethos-paper
# inspired by Andrew Karpathy's minGPT
# https://www.youtube.com/watch?v=kCc8FmEb1nY


import inspect
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, attention_weights: Optional[list] = None):
        super().__init__()
        assert config.hidden_size % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.hidden_size = config.hidden_size
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash or attention_weights is not None:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
                persistent=False,
            )
        self.attention_weights = attention_weights

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (hidden_size)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.hidden_size, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash and self.attention_weights is None:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            self.attention_weights.append(att.detach().cpu())
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.hidden_size, config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, attention_weights: Optional[list] = None):
        super().__init__()
        self.ln_1 = LayerNorm(config.hidden_size, bias=config.bias)
        self.attn = CausalSelfAttention(config, attention_weights=attention_weights)
        self.ln_2 = LayerNorm(config.hidden_size, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Ethos(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        self.embeddings = EhrEmbeddings(config)
        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [Block(config, None) for _ in range(config.n_layer)]
                ),
                ln_f=LayerNorm(config.hidden_size, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # if we are using adaptive softmax, we need to also create the head for the tail of the distribution

        # init all weights
        self.apply(self._init_weights)
        #
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            age_ids: Optional[torch.Tensor] = None,
            segment_ids: Optional[torch.Tensor] = None,
            record_rank_ids: Optional[torch.Tensor] = None,
            domain_ids: Optional[torch.Tensor] = None,
            target: Optional[torch.Tensor] = None,
        ):
        X_embed = self.embeddings(
            input_ids=input_ids,
            age_ids=age_ids,
            segment_ids=segment_ids,
            record_rank_ids=record_rank_ids,
            domain_ids=domain_ids,
        )
        x = self.transformer.drop(X_embed)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if target is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                ignore_index=-1,
                reduction="none",
            )

            loss = loss.mean()
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss


class EthosForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.embeddings = EhrEmbeddings(config)
        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [Block(config, None) for _ in range(config.n_layer)]
                ),
                ln_f=LayerNorm(config.hidden_size, bias=config.bias),
            )
        )
        self.lm_head_for_clf = nn.Linear(config.hidden_size, 2, bias=False)
        # if we are using adaptive softmax, we need to also create the head for the tail of the distribution

        # init all weights
        self.apply(self._init_weights)
        #
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            age_ids: Optional[torch.Tensor] = None,
            segment_ids: Optional[torch.Tensor] = None,
            record_rank_ids: Optional[torch.Tensor] = None,
            domain_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            target: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
        X_embed = self.embeddings(
            input_ids=input_ids,
            age_ids=age_ids,
            segment_ids=segment_ids,
            record_rank_ids=record_rank_ids,
            domain_ids=domain_ids,
        )
        x = self.transformer.drop(X_embed)
        for block in self.transformer.h:
            x = block(x)
        
        # Get hidden state of last code
        x = self.transformer.ln_f(x) 
        x = x[:, -1, :] # [batch_size * 2]
        logits = self.lm_head_for_clf(x)

        loss = None
        if target is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (target.dtype == torch.long or target.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), target.squeeze())
                else:
                    loss = loss_fct(logits, target)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), target.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, target)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )

        