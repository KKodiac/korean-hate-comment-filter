import numpy as np

import pandas as pd
import csv
import tensorflow as tf
print("tf version: ", tf.__version__)

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

from sklearn import metrics
from sklearn.model_selection import train_test_split


from tokenization_kocharelectra import KoCharElectraTokenizer
from transformers import DistilBertTokenizer, TFDistilBertModel


def encode(sentences, tokenizer, max_length):
        encode = tokenizer.batch_encode_plus(sentences,
            return_token_type_ids=False, pad_to_max_length=True, max_length=max_length
        )
        encoding_id = np.array(encode['input_ids'])
        return encoding_id


if __name__ == '__main__':
    tpu = None
    strategy = tf.distribute.MirroredStrategy()

    # Configuration
    num_of_rep = strategy.num_replicas_in_sync
    BATCH_SIZE = 50 * num_of_rep
    EPOCHS = 1
    MAX_LEN = 192
    opt = Adam(lr=1e-5)
    loss = 'binary_crossentropy'
    metrics=['accuracy']

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    tokenizer_ko = KoCharElectraTokenizer.from_pretrained("monologg/kocharelectra-small-discriminator")

    # 학습 데이터를 읽어온다. 
    train = pd.read_csv('data/train.tsv', quoting=csv.QUOTE_NONE,sep='\t', error_bad_lines=False)
    # 기존 토크나이저로 인코딩 한다. 
    x_train = encode(train.content.values, tokenizer, max_length=MAX_LEN)
    y_train = train.label.values

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.20)

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat().shuffle(2048).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    )

    valid_dataset = (
        tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE)
    )

    with strategy.scope():
        transformer_layer = TFDistilBertModel.from_pretrained('distilbert-base-multilingual-cased') # Fine-Tuning 할 distilbert 로딩
        input_word_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_word_ids") # 입력층 생성
        sequence_output = transformer_layer(input_word_ids)[0] # DistilBERT의 트랜스포머 층에서 Sequence 출력 받음
        cls_token = sequence_output[:, 0, :] # 출력층으로 넘길 각 문장단으로 나오는 Classification 토큰 받는다
        out = Dense(1, activation='sigmoid')(cls_token) # 이진 분류를 위한 출력층 
        model = Model(inputs=input_word_ids, outputs=out) # 모델 생성
        model.compile(optimizer=opt,
                    loss=loss, metrics=metrics) # 모델 컴파일
    model.summary()

    train_history = model.fit(
        train_dataset,
        steps_per_epoch=x_train.shape[0] // BATCH_SIZE,
        validation_data=valid_dataset,
        epochs=EPOCHS
    )

    model.save('model.h5')

