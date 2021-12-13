import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.npyio import load
import pandas as pd
import csv
import tensorflow 
from pykospacing import Spacing

def load_data():
    train_df = pd.read_csv('data/train.tsv', quoting=csv.QUOTE_NONE,sep='\t', error_bad_lines=False)
    test_df = pd.read_csv('data/test.tsv', quoting=csv.QUOTE_NONE, sep='\t', error_bad_lines=False)
    eval_df = pd.read_csv('data/val.tsv', quoting=csv.QUOTE_NONE, sep='\t', error_bad_lines=False)
    return train_df, test_df, eval_df

def proper_spacing(text):
    spacing = Spacing()

    return spacing(text)


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