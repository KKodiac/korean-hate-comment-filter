import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.npyio import load
import pandas as pd
import csv
import tensorflow 
from pykospacing import Spacing

# 크롤링 된 데이터를 읽어 오는 함수
def load_data():
    train_df = pd.read_csv('data/train.tsv', quoting=csv.QUOTE_NONE,sep='\t', error_bad_lines=False)
    test_df = pd.read_csv('data/test.tsv', quoting=csv.QUOTE_NONE, sep='\t', error_bad_lines=False)
    eval_df = pd.read_csv('data/val.tsv', quoting=csv.QUOTE_NONE, sep='\t', error_bad_lines=False)
    return train_df, test_df, eval_df

# PyKoSpacing 사용해서 적절한 공백을 추가하는 함수
def proper_spacing(text):
    spacing = Spacing()

    return spacing(text)

# 텍스트를 정제하는 함수
# 크롤링된 load_data를 이용해서 적절한 공백을 추가하고 재정렬함
if __name__ == "__main__":
    train_df, test_df, eval_df = load_data()
    dataframe = {
        'content': [],
        'label': []
    }
    with open('data/train.tsv', 'w') as f:
        for i, text in enumerate(train_df['content']):
            dataframe['content'].append(proper_spacing(text))
            dataframe['label'].append(train_df['label'][i])
    
        df = pd.DataFrame(dataframe)
        df.to_csv(f, sep='\t', index=False)

    with open('data/test.tsv', 'w') as f:
        for i, text in enumerate(train_df['content']):
            dataframe['content'].append(proper_spacing(text))
            dataframe['label'].append(train_df['label'][i])
    
        df = pd.DataFrame(dataframe)
        df.to_csv(f, sep='\t', index=False)

    with open('data/val.tsv', 'w') as f:
        for i, text in enumerate(train_df['content']):
            dataframe['content'].append(proper_spacing(text))
            dataframe['label'].append(train_df['label'][i])
    
        df = pd.DataFrame(dataframe)
        df.to_csv(f, sep='\t', index=False)