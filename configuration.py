#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   config.py
@Time    :   2024/09/12 11:19:21
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   None
"""
from typing import Optional


class Config:
    # dataset configuration
    train_file: Optional[str] = "2024927_cls.csv"
    validation_file: Optional[str] = None
    validation_rate: Optional[float] = 0.05
    max_seq_length: Optional[int] = 512
    # single_label_classification, multi_label_classification, regression
    problem_type = "single_label_classification"
    # model configuration
    model_path: Optional[str] = "qwen2_0.5B_instruct"

    deepspeed_config = "ds_z0_config.json"
    # output
    output_path: Optional[str] = "output"
    checkpoint_every: Optional[int] = 100
    save_total_limit: Optional[int] = 10

    # parameters
    learning_rate: Optional[float] = 1e-4
    epochs: Optional[int] = 10
    micro_batch_size: Optional[int] = 8
    micro_eval_size: Optional[int] = 8
    accu_steps: Optional[int] = 1
    lr_scheduler_type = "cosine"

    warmup_steps = 10
    weight_decay = 0.1
    adam_beta1 = 0.999
    adam_beta2 = 0.95

    eval_steps: Optional[int] = 20
    log_every: Optional[int] = 5
    evaluation_strategy: Optional[str] = "steps"
