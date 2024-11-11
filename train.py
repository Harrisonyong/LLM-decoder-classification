#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2024/09/13 14:53:58
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   None
'''

import os
import sys
import pathlib
import numpy as np
project_path = os.path.abspath(".")
sys.path.insert(0, project_path)
from transformers import TrainingArguments, Trainer, EvalPrediction
from model import ClassficationModel
from datahelper import DataSplit
from configuration import Config

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)
def metrics_for_label(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    labels = p.label_ids
    rank0_print("prediction:",preds)
    rank0_print("label:",labels)
    total = 0
    correct = 0
    for p_y, label_y in zip(preds, labels):
        total += 1
        if p_y == label_y:
            correct += 1
    return {"acc": correct / total if total >0 else 0}

train_args = TrainingArguments(
    output_dir=Config.output_path,
    num_train_epochs=Config.epochs,
    per_device_train_batch_size=Config.micro_batch_size,
    per_device_eval_batch_size=Config.micro_eval_size,
    gradient_accumulation_steps=Config.accu_steps,
    eval_strategy=Config.evaluation_strategy,
    eval_steps=Config.eval_steps,
    logging_dir=Config.output_path,
    logging_steps=Config.log_every,
    save_steps=Config.checkpoint_every,
    save_total_limit=Config.save_total_limit,
    learning_rate=Config.learning_rate,
    lr_scheduler_type=Config.lr_scheduler_type,
    warmup_steps=Config.warmup_steps,
    weight_decay=Config.weight_decay,
    adam_beta1=Config.adam_beta1,
    adam_beta2=Config.adam_beta2,
    fp16=True,
    load_best_model_at_end=True,
    deepspeed=Config.deepspeed_config,
    report_to="none",
)

class ClassificationTrainer():
    def __init__(self) -> None:
        claification_model = ClassficationModel()
        self.tokenizer = claification_model.tokenize()
        rank0_print("tokenizer initialized")
        self.datasets = DataSplit(self.tokenizer)
        self.train_datset, self.val_datset = self.datasets()
        # print("val_dataset_after_token", self.val_datset[:])
        self.label_list = self.datasets.label_list
        self.id2label = self.datasets.id2label
        self.label2id = self.datasets.label2id
        rank0_print(f"data loaded {len(self.train_datset), len(self.val_datset)}")
        rank0_print("label_list:", self.label_list)
        rank0_print("label2id", self.label2id)
        self.model = claification_model.model(len(self.label_list))
        self.model.config.id2label = self.id2label
        self.model.config.label2id = self.label2id
        rank0_print("num_labels:",self.model.config.num_labels)
        rank0_print("model loaded")
    
    def train(self):
        rank0_print(self.train_datset[:5]["text"], self.train_datset[:5]["label"])
        rank0_print(self.val_datset[:]["text"], self.val_datset[:]["label"])

        trainer = Trainer(
                model=self.model,
                args=train_args,
                train_dataset=self.train_datset,
                eval_dataset=self.val_datset,
                tokenizer=self.tokenizer,
                compute_metrics=metrics_for_label,
            )
        if list(pathlib.Path(train_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        trainer.save_state()
        final_dir = train_args.output_dir + "/endpoint"
        trainer.save_model(final_dir)
        self.tokenizer.save_pretrained(final_dir)

def main():
    global local_rank
    local_rank = train_args.local_rank
    ctrainer = ClassificationTrainer()
    ctrainer.train()

if __name__ == '__main__':
    main()
    