from typing import Optional, Union, Tuple, List
import torch
import torch.nn as nn
from torch.autograd import Function
from transformers import (BertForMaskedLM, BertForPreTraining,
                          BertForSequenceClassification)
from transformers.models.bert.modeling_bert import (
    BertForMaskedLM, MaskedLMOutput, CrossEntropyLoss, BertModel,
    BaseModelOutputWithPoolingAndCrossAttentions, MSELoss,
    MaskedLMOutput,
    BertEncoder, BertPooler, SequenceClassifierOutput, 
    SequenceClassifierOutput,
    MSELoss, BCEWithLogitsLoss, BertForPreTrainingOutput, BertLayer,
    BertAttention, BertIntermediate, BertOutput,
    )
from transformers import BertModel, BertPreTrainedModel
import torch.nn.functional as F

import sys
from pathlib import Path
abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)
from src.embeddings import EhrEmbeddings
from src.adapter import AdapterEmbeddedBertEncoder, BottleneckAdapter


class BertModelLabelAdded(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
 
        self.embeddings = EhrEmbeddings(config)
        if config.problem_type == 'single_label_classification' and config.use_adapter:
            self.encoder = AdapterEmbeddedBertEncoder(config)
        else:
            self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

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
            # smoothing_loss = loss_mse(sequence_output, representations)
            # if self.config.smooth:
            #     masked_lm_loss += 0.1 * smoothing_loss

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
            # smoothing_loss = loss_mse(sequence_output, representations)
            # if self.config.smooth:
            #     total_loss += 0.1 * smoothing_loss

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

        _pooled_output = outputs[1]
        pooled_output = self.dropout(_pooled_output)
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
                loss = loss_fct(logits, target.float())
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=_pooled_output, # Pooled output for feature
            attentions=outputs.attentions,
        )

import random
import copy
class MixLayer(BertLayer):
    def __init__(
            self, 
            attention_layers, 
            intermediate_layers, 
            output_layers, 
            config):
        super().__init__(config)
        self.attention_layers = nn.ModuleList(attention_layers)
        self.intermediate_layers = nn.ModuleList(intermediate_layers)
        self.output_layers = nn.ModuleList(output_layers)
        self.config = config
    
    def select(self, j, k):
        self.attention = self.attention_layers[j]
        self.intermediate = self.intermediate_layers[k]
        self.output = self.output_layers[k]

    def _ensemble(self, modules: nn.ModuleList) -> nn.Module:
        avg_module = copy.deepcopy(modules[0])
        
        with torch.no_grad():
            for name, param in avg_module.named_parameters():
                stacked = torch.stack([m.state_dict()[name] for m in modules])
                mean_param = torch.mean(stacked, dim=0)
                param.copy_(mean_param)

        return avg_module

    def ensemble(self):
        self.attention = self._ensemble(self.attention_layers)
        self.intermediate = self._ensemble(self.intermediate_layers)
        self.output = self._ensemble(self.output_layers)



class MixEHR(BertForSequenceClassification):
    def __init__(self, config, **baseline_models):
        super().__init__(config)
        self.config = config        
        self.mix_layers = []

        for layer_idx in range(config.num_hidden_layers):
            attention_layers = []
            intermediate_layers = []
            output_layers = []

            for g, prev_model in baseline_models.items():
                _layer = BertAttention(config)
                _layer.load_state_dict(prev_model.bert.encoder.layer[layer_idx].attention.state_dict())
                attention_layers.append(_layer)

                _layer = BertIntermediate(config)
                _layer.load_state_dict(prev_model.bert.encoder.layer[layer_idx].intermediate.state_dict())
                intermediate_layers.append(_layer)

                _layer = BertOutput(config)
                _layer.load_state_dict(prev_model.bert.encoder.layer[layer_idx].output.state_dict())
                output_layers.append(_layer)
            
            self.mix_layers.append(MixLayer(attention_layers, intermediate_layers, output_layers, config))
        
        self.mix_layers = nn.ModuleList(self.mix_layers)
        self.embeddings = prev_model.bert.embeddings

        
        self.pooler = BertPooler(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels
        
        for name, p in self.embeddings.named_parameters():
            if 'concept_embedding' in name:
                p.requires_grad_(False)

        self.kldiv = nn.KLDivLoss(reduction='batchmean')
        self.softmax = nn.Softmax(dim=1)
                
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
        inference=False,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
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
        past_key_values_length = 0

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

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embeddings(
            input_ids=input_ids,
            age_ids=age_ids,
            segment_ids=segment_ids,
            record_rank_ids=record_rank_ids,
            domain_ids=domain_ids,
            inputs_embeds=inputs_embeds,
        )
        
        hidden_states, hidden_states_p = embedding_output, embedding_output
        if inference:
            for layer_module in self.mix_layers:
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = None
                past_key_value = None

                layer_module.ensemble()
                layer_outputs = layer_module(
                    hidden_states,
                    extended_attention_mask,
                    layer_head_mask,
                    past_key_value,
                    output_attentions,
                )
                hidden_states = layer_outputs[0]
                if use_cache:
                    next_decoder_cache += (layer_outputs[-1],)
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

        else:
            for layer_module in self.mix_layers:
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = None
                past_key_value = None

                j, jp = random.sample(range(self.config.num_experts), 2)
                k, kp = random.sample(range(self.config.num_experts), 2)
                layer_module.select(j, k)
                layer_outputs = layer_module(
                    hidden_states,
                    extended_attention_mask,
                    layer_head_mask,
                    past_key_value,
                    output_attentions,
                )
                layer_module.select(jp, kp)
                layer_outputs_p = layer_module(
                    hidden_states_p,
                    extended_attention_mask,
                    layer_head_mask,
                    past_key_value,
                    output_attentions,
                )
                hidden_states = layer_outputs[0]
                hidden_states_p = layer_outputs_p[0]
                if use_cache:
                    next_decoder_cache += (layer_outputs[-1],)
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
            
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        _pooled_output = self.pooler(hidden_states)
        pooled_output = self.dropout(_pooled_output)
        logits = self.classifier(pooled_output)

        if not inference:
            _pooled_output_p = self.pooler(hidden_states_p)
            pooled_output_p = self.dropout(_pooled_output_p)
            logits_p = self.classifier(pooled_output_p)

            klloss1 = self.kldiv(self.softmax(logits).log(), self.softmax(logits_p))
            klloss2 = self.kldiv(self.softmax(logits_p).log(), self.softmax(logits))

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
                
            if not inference:
                loss = loss + 0.5 * (klloss1 + klloss2)
                
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=_pooled_output, # Pooled output for feature
        )
