from keras.models import  Model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator, text_segmentate, to_array
from preprocess_and_finetune_model import get_data
from bert4keras.models import build_transformer_model
import numpy as np
from tqdm import tqdm

maxlen = 256
batch_size = 30

config_path = 'dataset/cased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'dataset/cased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'dataset/cased_L-12_H-768_A-12/vocab.txt'

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

data = get_data(truth_path,text_path)

train_generator = data_generator(data[:36821], batch_size)
test_generator = data_generator(data[36821:], batch_size)

bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    # with_pool=True,
    return_keras_model=False,
    num_hidden_layers=12,
)
output1 = Lambda(lambda x: x[:, 0])(bert.model.output)
output = Dropout(rate=0.2)(output1)
output = Dense(units=2, activation='softmax', )(output)

model = Model(bert.model.inputs, output)

cls_layer = Model(inputs=model.input, outputs=model.layers[-3].output)  #后面会从cls_layer提取cls向量作为句子表示


def get_sents_represent(generator, texts_num):
    sents_r = np.array([])
    for x_true, y_true in tqdm(generator):
        sent_r = cls_layer.predict(x_true)
        sents_r = np.append(sents_r, sent_r)

    sents_rs = sents_r.reshape(texts_num, 30, 768)

    return sents_rs


train_sents_rs = get_sents_represent(train_generator, len(data[:36821]))
test_sents_rs = get_sents_represent(test_generator, len(data[36821:]))

np.save("train_sents_rs.npy",train_sents_rs)
np.save("test_sents_rs.npy",test_sents_rs)

