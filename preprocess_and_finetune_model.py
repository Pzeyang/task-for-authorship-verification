import random
import numpy as np
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, AdaFactorV1
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Dropout, Dense, Lambda, LSTM, GlobalAveragePooling1D
from bert4keras.snippets import  text_segmentate
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json

def get_data(truth_path, text_path):

    truth=[]
    with open(truth_path,'r') as f:
        for l in f:
            data = json.loads(l)
            truth.append((data['id'], data['same'], data['authors']))

    index=0

    with open(text_path,'r') as f:
        datas=[]
        for l in tqdm(f):
            data = json.loads(l)
            if truth[index][0]==data['id']:

                text1 = text_segmentate(data['pair'][0], maxlen=510, seps='.?!')
                text2 = text_segmentate(data['pair'][1], maxlen=510, seps='.?!')

                while len(text1) < 30 or len(text2) < 30:
                    if len(text1) < 30:
                        n_text1 = []
                        for i in range(30):
                            for sent in text1:
                                n_text1.append(sent)
                        text1 = n_text1
                    elif len(text2) < 30:
                        n_text2 = []
                        for i in range(30):
                            for sent in text2:
                                n_text2.append(sent)
                        text2 = n_text2

                datas.append((text1, text2, int(truth[index][-2]), str(data['id']), truth[index][-2], truth[index][-1]))

            index+=1


    return datas

truth_path = 'D:\PAN\pan20-authorship-verification-training-small\dataset\pan20-authorship-verification-training-small-truth.jsonl'
text_path = 'D:\PAN\pan20-authorship-verification-training-small\dataset\pan20-authorship-verification-training-small.jsonl'
data = get_data(truth_path,text_path)
data = data[:36821] #这里是取出70%当训练集进行微调

maxlen = 256
batch_size = 30

config_path = 'D:\PAN\dataset\cased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'D:\PAN\dataset\cased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'D:\PAN\dataset\cased_L-12_H-768_A-12/vocab.txt'

 # 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=False)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label, id, _, _) in self.sample(random):
            for index, sent in enumerate(text1[:30]):
                token_ids, segment_ids = tokenizer.encode(text1[index], text2[index], maxlen=maxlen)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([label])
                if len(batch_labels) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = sequence_padding(batch_labels)

                    yield [batch_token_ids, batch_segment_ids], batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []

generator = data_generator(data, batch_size)

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):

        model.save_weights('/home/peng21/home/'+str(epoch) +'_model.weights')

evaluator = Evaluator()

bert0 = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
    num_hidden_layers=12,
)

output1 = Lambda(lambda x: x[:, 0])(bert0.model.output)
output = Dropout(rate=0.2)(output1)
output = Dense(units=2, activation='softmax', )(output)

model = keras.models.Model(bert0.model.inputs, output)

model.compile(
    optimizer=Adam(2e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
model.fit(
    generator.forfit(),
    steps_per_epoch=len(generator),
    epochs=5,
    callbacks=[evaluator]
)

