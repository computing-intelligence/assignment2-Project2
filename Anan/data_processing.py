# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 17:53:55 2018

@author: Anan
"""
import numpy as np
import pickle
from collections import Counter
import random
import tensorflow as tf
 
max_comment_length = 200
split_frac = 0.8  
num_labels = 3

def is_CN_char(ch):
    return ch >= u'\u4e00' and ch <= u'\u9fa5'

def cut(string):
    return list(jieba.cut(string))

def get_stopwords(filename = "D:/Code/Pycharm/Data_source/chinese_stopwords.txt"):
    stopwords_dic = open(filename, encoding= 'utf-8')
    stopwords = stopwords_dic.readlines()
    stopwords = [w.strip() for w in stopwords]
    stopwords_dic.close()
    return stopwords

def convert2simple(word):
    openCC = OpenCC('tw2sp')
    return openCC.convert(word)

def clean_sentence(sentence):
    stopwords = get_stopwords()
    sentence = ''.join(filter(is_CN_char,sentence))
    sentence = convert2simple(sentence)
    words = [w for w in cut(sentence) if len(w)>1 and w not in stopwords]
    words = ' '.join(words)
    return words   

def save_txt(data,filename):
    with open(filename,"wb") as f:
        pickle.dump(data,f)
                
def get_comment_each_star():
    content = pd.read_csv('douban_movie_comments.csv',encoding='gb18030')
    content['comment'] = content['comment'].fillna('')
    content['comment'] = content['comment'].apply(clean_sentence)
    for i in range(1,6):
        comment = [content.iloc[j].comment for j in range(len(content)) if content.iloc[j].star == i]
        comment = [c for c in comment if len(c)!=0]
        comment_lens = Counter([len(x) for x in comment])
        if comment_lens[0] == 0:
            save_txt(comment,"comments_star{}.txt".format(i))

def read_txt(filename):
    with open(filename, "rb") as fp:
           b = pickle.load(fp)
    return b

def get_random_comment(data,length=6000):
    return [data[i] for i in random.sample(range(len(data)),length)]

def get__binary_comment(star,length):
    comment = read_txt("comments_star{}.txt".format(star))
    comment = get_random_comment(comment,length=length)
    return comment

def word_to_id(vocab):
    counts = Counter(vocab)
    vocab = sorted(counts, key=counts.get, reverse=True)
    word_to_id = { word : i for i, word in enumerate(vocab)}
    id_to_word = {i:word for i,word in enumerate(vocab)}
    return word_to_id, id_to_word

def comment_to_id(word_to_id,comments):
    comment_to_id = []
    for comment in comments:
        comment_to_id.append([word_to_id[word] for word in comment.split()] )
    return comment_to_id

def pad_sequences(comment_to_id,maxlen,padding='post',truncating='post'):
    features = np.zeros((len(comment_to_id), maxlen), dtype=int)
    for i,comment in enumerate(comment_to_id):
        if len(comment) <= maxlen and padding == 'pre':
            features[i, -len(comment):] = np.array(comment)[:maxlen]
        if len(comment) <= maxlen and padding == 'post':
            features[i, :len(comment)] = np.array(comment)[:maxlen]
        if len(comment) > maxlen and truncating == 'post':
            features[i, :] = np.array(comment)[:maxlen]
        if len(comment) > maxlen and truncating == 'pre':
            features[i, :] = np.array(comment)[len(comment)-maxlen:]           
    return features
      
def split_dataset(pad_comments,labels):
    split_index = int(len(pad_comments)*split_frac)
    data_list = list(zip(pad_comments, labels))
    random.shuffle(data_list)
    pad_comments, labels = zip(*data_list)
    x_train, x_test = pad_comments[:split_index], pad_comments[split_index:]
    y_train, y_test = labels[:split_index], labels[split_index:]
    return x_train,y_train,x_test,y_test          


comment_star1 = get__binary_comment(1,10000)   
comment_star2 = get__binary_comment(2,5000) 
comment_star3 = get__binary_comment(3,14500)   
negetive_comment = comment_star1 + comment_star2
positive_comment=  get__binary_comment(5,15000) 
all_comment = positive_comment + comment_star3 + negetive_comment
labels = [2]*15000 + [1]*14500 + [0]*15000


labels = (np.arange(num_labels) == np.array(labels)[:,None]).astype(np.float32)
vocab = ' '.join(all_comment).split()
vocab.append('unknown')
word_to_id, id_to_word = word_to_id(vocab)
comment_to_id = comment_to_id(word_to_id,all_comment)
pad_comments = pad_sequences(comment_to_id,maxlen=max_comment_length,padding='post',truncating='post')

x_train,y_train,x_test,y_test = split_dataset(pad_comments,labels)

save_txt(x_train,"x_train_sentiment")
save_txt(y_train,"y_train_sentiment")
save_txt(x_test,"x_test_sentiment")
save_txt(y_test,"y_test_sentiment") 
save_txt(y_test,"y_test_sentiment") 
save_txt(word_to_id,"word_to_id_sentiment") 


