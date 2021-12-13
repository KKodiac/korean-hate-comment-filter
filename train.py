# Import to list directories 
import os

print(os.getcwd())
print(os.listdir(os.getcwd()))

### import for time and time monitoring 
import time
## Imports for Metrics operations 
import numpy as np 
## Imports for dataframes and .csv managment
import pandas as pd 
import csv
# TensorFlow library  Imports
import tensorflow as tf
print("tf version: ", tf.__version__)

# Keras (backended with Tensorflow) Imports
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import CSVLogger

# Sklearn package for machine learining models 
## WILL BE USED TO SPLIT  TRAIN_VAL DATASETS
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Garbage Collectors
import gc
import sys

## Install Transformers 
os.system('pip install transformers==4.13.0')



# Import Transformers to get Tokenizer for bert and bert models

import transformers
from transformers import DistilBertTokenizer, TFDistilBertModel 

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver();
    tf.config.experimental_connect_to_cluster(tpu);
    tf.tpu.experimental.initialize_tpu_system(tpu);
    strategy = tf.distribute.TPUStrategy(tpu);
except ValueError:
    tpu = None
    strategy = tf.distribute.MirroredStrategy()

# Configuration
num_of_rep = strategy.num_replicas_in_sync
BATCH_SIZE = 50 * num_of_rep
EPOCHS = 2
MAX_LEN = 192
opt = Adam(lr=1e-5)
loss = 'binary_crossentropy'
metrics=['accuracy']

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
def encode(sentences, tokenizer, max_length):
    encode = tokenizer.batch_encode_plus(sentences,
        return_token_type_ids=False, pad_to_max_length=True, max_length=max_length
    )
    encoding_id = np.array(encode['input_ids'])
    return encoding_id

train = pd.read_csv('data/train.tsv', quoting=csv.QUOTE_NONE,sep='\t', error_bad_lines=False)
x_train = encode(train.content.values, tokenizer, max_length=MAX_LEN)
y_train = train.label.values

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, 
                                                      test_size=0.20)

train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat().shuffle(2048).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE)
)

with strategy.scope():
    transformer_layer = TFDistilBertModel.from_pretrained('distilbert-base-multilingual-cased')# get pre-trained bert transformer
    input_word_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_word_ids") # Keras Input layer 
    sequence_output = transformer_layer(input_word_ids)[0] # Get the seq output from Transformer layer provided
    cls_token = sequence_output[:, 0, :] # get cls tokkens from each scentence to feed to output
    out = Dense(1, activation='sigmoid')(cls_token) # create output layer with one node <Binary Classification>
    model = Model(inputs=input_word_ids, outputs=out) # Create Model
    model.compile(optimizer=opt, 
                  loss=loss, metrics=metrics) # Compile model
model.summary()

train_history = model.fit(
    train_dataset,
    steps_per_epoch=x_train.shape[0] // BATCH_SIZE,
    validation_data=valid_dataset,
    epochs=EPOCHS
)

model.save('my_hate_model.h5')