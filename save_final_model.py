import numpy as np

from preprocess_and_finetune_model import get_data
data = get_data(truth_path,text_path)

labels=[]
for i in data:
    labels.append(i[2])

train_sents_r = np.load("train_sents_rs.npy",mmap_mode = 'r')


train_labels=np.array(labels[:36821])
valid_sents_r = np.load("test_sents_rs.npy",mmap_mode = 'r')
valid_labels=np.array(labels[36821:])

from keras.layers import Dense,GlobalAveragePooling1D
from keras.models import Sequential

model = Sequential()
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu', ))
model.add(Dense(units=2, activation='softmax', ))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_sents_r,train_labels,epochs=500,validation_data=(valid_sents_r,valid_labels))
model.save('final_model.h5')