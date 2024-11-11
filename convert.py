#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   convert.py
@Time    :   2024/09/18 10:54:21
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   None
'''

import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel, AutoConfig

ORIGINAL_MODEL_PATH = '/home/mai-llm-train-service/yql/wedoctor_classfication/output/endpoint'

config = AutoConfig.from_pretrained(
        ORIGINAL_MODEL_PATH,
        trust_remote_code=True,
        )
tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_PATH, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(ORIGINAL_MODEL_PATH)

model_state_dict = model.state_dict()
torch.save(model, '/home/mai-llm-train-service/yql/wedoctor_classfication/version/wedoctor-930-v2/wedoctor_cls_930.pth')
tokenizer.save_pretrained('/home/mai-llm-train-service/yql/wedoctor_classfication/version/wedoctor-930-v2')