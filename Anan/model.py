# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 20:00:23 2018

@author: Anan
"""

import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Bidirectional, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras import regularizers
import matplotlib.pyplot as plt
 
max_comment_length = 200
embedding_vecor_length = 256

def read_txt(filename):
    with open(filename, "rb") as fp:
           b = pickle.load(fp)
    return b

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

word_to_id = read_txt("word_to_id_sentiment")
x_train_binary = read_txt('x_train_sentiment')
y_train_binary = read_txt('y_train_sentiment')
x_train = np.array(x_train_binary)
y_train = np.array(y_train_binary)   


model = Sequential()
model.add(Embedding(len(word_to_id), embedding_vecor_length, input_length=max_comment_length, dropout=0.2))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(50, return_sequences=True)))
model.add(Bidirectional(LSTM(50)))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax', kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
history = model.fit(x_train, y_train,validation_split=0.20, epochs=5,verbose=1, batch_size=1000)


print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

x_test = np.array(read_txt('x_test_sentiment'))
y_test = np.array(read_txt('y_test_sentiment'))
preds = model.predict(x_test)
print(accuracy(preds,y_test))
