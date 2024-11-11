#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   inference.py
@Time    :   2024/09/12 10:44:21
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   None
'''
import time
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification, AutoModel
from model import CustomerClassfication

ORIGINAL_MODEL_PATH = '/home/mai-llm-train-service/yql/wedoctor_classfication/output/endpoint'


tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_PATH, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(ORIGINAL_MODEL_PATH, trust_remote_code=True)
# model = torch.load("/home/mai-llm-train-service/yql/wedoctor_classfication/version/0918_2/wedoctor_cls.pth", map_location="cpu")

start = time.time()
# 准备输入文本
texts = [
  "是否需要患者进行长期监控？",
  "此次随访后，提交健康评估结果给健康顾问"
]
# 创建标签到索引的映射
label_to_id = model.config.label2id
id_to_label = model.config.id2label
print(model)
# 对文本进行编码
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# 进行推理
with torch.no_grad():
    outputs = model(**inputs)

# 获取预测结果
logits = outputs[0]
v, i= torch.topk(logits, k=2, dim=-1)
predictions = i.tolist()

# 打印预测结果

print(id_to_label)
print(predictions)
for i in range(len(texts)):
    text = texts[i]
    prediction = predictions[i]
    print(f"文本: {text} -> 预测类别: {[id_to_label[int(s)] for s in prediction]}")
print(time.time() - start)
