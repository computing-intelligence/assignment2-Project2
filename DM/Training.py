import pandas as pd
import numpy as np
import jieba
from collections import Counter

#读取本地保存的豆瓣爬虫数据，提取text和rating值并切分正面和负面评论
df_all=pd.read_csv('C:/Users/trans02/PycharmProjects/untitled1/pycode/Sentiment Analysis/text_rating.csv',encoding='gb18030')
comment_negative_ls=[]
comment_positive_ls=[]
for k in range(len(df_all['text'])):#遍历所有评论条目，将正面和负面评论存入对应list中
	if df_all.loc[k,'rating']==1 or df_all.loc[k,'rating']==2:comment_negative_ls.append(df_all.loc[k,'text'])
	if df_all.loc[k,'rating']==4 or df_all.loc[k,'rating']==5:comment_positive_ls.append(df_all.loc[k,'text'])

def cut(string):
	return list(jieba.cut(str(string)))

def wash(str_original):
	# 要清洗删除的字符
	ls_delete = ['\u3000', '\\n', ',', '。', ':', '：', '，', '“', '”', '（', '）', '《', '》', '！', '!', '、', '(', ')', '·']
	for k in range(len(ls_delete)):
		str_original=str(str_original).replace(ls_delete[k],' ')#用空格替代，相当于标点符号处都固定分词，免得后面分词混淆
	return str_original

#分词和构建label
comment_NP_ls=comment_negative_ls+comment_positive_ls
comment_NP_ls_cut=[' '.join(cut(s)) for s in [wash(t) for t in comment_NP_ls]]
comment_NP_cut_sum=''
for k in comment_NP_ls_cut:comment_NP_cut_sum+=(k+' ')
counts=Counter(comment_NP_cut_sum.split())
vocab_size=len(counts)
labels =[0]*len(comment_negative_ls)+[1]*len(comment_positive_ls)

#one_hot编码
from keras.preprocessing.text import one_hot
encoded_docs = [one_hot(d, vocab_size) for d in comment_NP_ls_cut]

#设置一条评论的最大长度（含词数），多的截断，少的补0。实际评论的平均长度为30
max_length = 30
vector_size=300
from keras.preprocessing.sequence import pad_sequences
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post',truncating='post')

#feature和label压缩后乱序，再解压
zipped_docs_labels = list(zip(padded_docs, labels))
import random
random.shuffle(zipped_docs_labels)
padded_docs, labels = zip(*zipped_docs_labels)
padded_docs=np.array(padded_docs)
labels=list(labels)

#准备训练集和测试集数据
train_ratio=0.9#训练集占全部样本的比例，供切分
train_amount=int(train_ratio*len(labels))#len(labels)是全体样本的个数
train_x=padded_docs[:train_amount]
train_y=labels[:train_amount]
test_x=padded_docs[train_amount:]
test_y=labels[train_amount:]

#模型构建
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Flatten,Dense
from keras.layers import Conv1D,LSTM,Dropout,Bidirectional
from keras.layers.convolutional import MaxPooling1D
from keras import regularizers

model = Sequential()
model.add(Embedding(vocab_size, vector_size, input_length=max_length))
model.add(Conv1D(filters=30, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(50, return_sequences=True)))
# model.add(Bidirectional(LSTM(50)))
model.add(Dropout(0.5))
model.add(Flatten())
# model.add(Dense(1, activation='softmax', kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)))
model.add(Dense(1, activation='sigmoid'))

#模型编译
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())#打印模型信息

#模型拟合
history=model.fit(train_x, train_y, validation_split=0.2,epochs=5, verbose=1)

import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.title('model accuracy')
plt.ylabel('value')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss'])
plt.show()

#模型准确度评估
loss, accuracy = model.evaluate(test_x, test_y,  verbose=1)
print('Accuracy: %f' % (accuracy*100))
