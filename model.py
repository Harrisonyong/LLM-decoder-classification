#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2024/09/12 16:28:03
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   None
'''
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from typing import Optional
from configuration import Config
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, AutoModel, PreTrainedModel, Qwen2Config, Qwen2Model
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

class CustomerClassfication(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = AutoModel.from_pretrained(Config.model_path, 
                                               config = config,
                                               trust_remote_code=True)
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.classifier_layers = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier_mid_layers = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.activation_layers = nn.Sigmoid()
        self.classifier = nn.Linear(config.hidden_size // 2, config.num_labels)
        self.post_init()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        pooled_output = self.classifier_layers(hidden_states)
        pooled_output = self.classifier_mid_layers(pooled_output)
        pooled_output = self.activation_layers(pooled_output)
        logits = self.classifier(pooled_output)
        
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths] 
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class ClassficationModel:
    def __init__(self):
        self.model_path = Config.model_path
    def tokenize(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        return tokenizer
    def model(self, num_labels):
        self.config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            output_hidden_states=True,
            pad_token_id=151643,
            finetuning_task="text-classification",
            num_labels=num_labels,
            problem_type=Config.problem_type
            )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            config=self.config,
            trust_remote_code=True,
        )
        return model
    
        
# if __name__ == '__main__':
#     config = AutoConfig.from_pretrained(
#             Config.model_path,
#             trust_remote_code=True,
#             output_hidden_states=True,
#             pad_token_id=151643,
#             num_labels=16,
#             problem_type=Config.problem_type
#             )
#     model = CustomerClassfication(config=config)
#     print(model)
#     print(model.base_model_prefix)
    