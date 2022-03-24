import pandas as pd
import torch
import numpy as np
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification, TFDistilBertModel, DistilBertConfig
from tensorflow import keras

tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-hate-speech")
config = DistilBertConfig.from_json_file('config.json')
pre_trained_model = AutoModelForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-hate-speech")
model = TFAutoModelForSequenceClassification.from_pretrained("model.h5", config=config)
distil_model = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased')

# koelectra-base-v3-hate-speech 로 이미 학습된 모델의 예측 결과를 출력
def predict_with_pretrained(text):
    with torch.no_grad():
        input = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt")
        outputs = pre_trained_model(**input)
        pred = outputs.logits.argmax(dim=1).item()
        print("pre-pred", pred)
        return pred

# distilbert-base-multilingual-cased 를 기반으로 재학습 시킨 모델의 예측 결과를 출력
def predict(text):
    with torch.no_grad():
        input = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt")
        input = {'input_ids': input['input_ids'].numpy(),
                    'attention_mask': input['attention_mask'].numpy()}
        output = model(**input, return_dict=True)

        pred = np.argmax(output.logits, axis=1)
        
        print("pred", pred)
        return pred


def distil_predict(text):
    with torch.no_grad():
        input = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt")
        input = {'input_ids': input['input_ids'].numpy(),
                    'attention_mask': input['attention_mask'].numpy()}
        output = distil_model(**input, return_dict=True)

        pred = np.argmax(output.logits, axis=1)
        
        print("pred", pred)
        return pred