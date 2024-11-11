#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   datahelper.py
@Time    :   2024/09/12 14:07:51
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   None
'''

from typing import List
from dataclasses import dataclass
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from configuration import Config as config
    

@dataclass
class DataSplit:
    tokenizer: "PreTrainedTokenizer"
    def __post_init__(self):
        print(config.problem_type)
        self.data_path = config.train_file
        self.max_seq_length = min(config.max_seq_length, self.tokenizer.model_max_length)
        if config.validation_file:
            self.data_files = {"train": config.train_file, "validation": config.validation_file}
            dataset_list = self.load(self.data_files, split=["train", "validation"])
            self.train_dataset, self.val_dataset = dataset_list
            self.datasets = self.train_dataset
        else:
            self.data_files = config.train_file
            self.datasets = self.load(self.data_files)["train"]
            split_datasets = self.datasets.train_test_split(test_size=config.validation_rate, seed=122)
            self.train_dataset = split_datasets["train"]
            self.val_dataset = split_datasets["test"]
        self.get_label_list()
    
    def load(self, data_files, split="train", file_format="csv"):
        if isinstance(data_files, str):
            return load_dataset(file_format, data_files=data_files)
        else:
            return load_dataset(file_format, data_files=data_files, split=split)
    def get_label_list(self):
        label_list = self.datasets.unique("label")
        self.label_list = sorted(label_list)
        self.label_list.append("其他")
        self.label2id = {v:i for i, v in enumerate(self.label_list)}
        self.id2label = {id:label for label, id in self.label2id.items()}
    
    def multi_labels_to_ids(self, labels) -> List[float]:
        ids = [0.0] * len(self.label_list)
        if isinstance(labels, str):
            lable_list = labels.split("、")
            for label in lable_list:
                ids[self.label2id[label]] = 1.0
        elif isinstance(labels, list):
            for label in labels:
                ids[self.label2id[label]] = 1.0
        return ids
    def preprocess_function(self, examples):
        result = self.tokenizer(examples["text"], padding="max_length", max_length=self.max_seq_length)
        if config.problem_type == "single_label_classification":
            result["label"] = [self.label2id[label] for label in examples["label"]]
        elif config.problem_type == "multi_label_classification":
            result["label"] = [self.multi_labels_to_ids(label) for label in examples["label"]]
        return result
    
    def __call__(self):
        return self.train_dataset.map(self.preprocess_function, batched=True), self.val_dataset.map(self.preprocess_function, batched=True)