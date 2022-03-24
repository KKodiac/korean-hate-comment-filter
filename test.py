import pandas as pd
import numpy as np
import csv

from transformers.models import distilbert

from train import encode

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tensorflow import keras
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, TFDistilBertModel, DistilBertConfig
from sklearn.metrics import accuracy_score
distil_model = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased')
MAX_LEN = 192

config = DistilBertConfig.from_json_file('config.json')
model = TFAutoModelForSequenceClassification.from_pretrained("model.h5", config=config)
tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-hate-speech")


# DistilBERT-Multilingual에서 자체 fine-tuning한 모델 테스팅
def my_test_set():
    test = pd.read_csv('data/test.tsv', quoting=csv.QUOTE_NONE,sep='\t', error_bad_lines=False)
    x_test = encode(test.content.tolist(), tokenizer, MAX_LEN)
    test_dataset = (tf.data.Dataset.from_tensor_slices(x_test).batch(32))
    pr = model.predict(test_dataset, verbose=1)
    result = np.argmax(pr.logits, axis=1)
    y_pred_testset_kaggle = result
    y_true_testset_kaggle = test["label"].tolist()
    
    print(np.argmax(pr.logits, axis=1))
    print(accuracy_score(y_true_testset_kaggle, y_pred_testset_kaggle))


# DistilBERT-Multilingual 테스팅
def distil_test_set():
    test = pd.read_csv('data/test.tsv', quoting=csv.QUOTE_NONE,sep='\t', error_bad_lines=False)
    x_test = encode(test.content.tolist(), tokenizer, MAX_LEN)
    test_dataset = (tf.data.Dataset.from_tensor_slices(x_test).batch(32))
    pr = distil_model.predict(test_dataset, verbose=1)
    print(pr)
    result = np.argmax(pr.logits, axis=1)
    print(result)
    y_pred_testset_kaggle = result
    y_true_testset_kaggle = test["label"].tolist()
    print(accuracy_score(y_true_testset_kaggle, y_pred_testset_kaggle))

    
if __name__ == '__main__':
    distil_test_set()
    my_test_set()