# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 13:42:18 2017
@author: Junior

简易语种检测器_朴素贝叶斯分类

"""
import codecs
import os
os.chdir(r'F:\付费学习\破解\机器学习全套课程\codec_myself\08')
os.getcwd()

### 导入数据
data_path = r'.\Lecture_2\Language-Detector\data.csv'
# way1
in_f = open(data_path,encoding = 'utf-8')
lines = in_f.readlines()
in_f.close()
# way2
with codecs.open(data_path,'r','utf-8') as fp:
    lines = fp.readlines()
    
'''
 a  b  c  d  e
-5 -4 -3 -2 -1
'''
dataset = [(line.strip()[:-3],line.strip()[-2:]) for line in lines]

### sklearn划分测试与训练数据集
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
x,y = zip(*dataset)
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state =1)

### 正则表达式去除噪声数据
import re
def remove_noise(document):
    noise_pattern =re.compile("|".join(["http/S+","\@\w+","\#\w+"]))
    clean_text = re.sub(noise_pattern,"",document)
    return clean_text

### 抽取特征
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(
    lowercase = True,
    analyzer = 'char_wb', # tokenise by character ngrams
    ngram_range = (1,2),
    max_features = 1000,
    preprocessor = remove_noise
)
vec.fit(x_train)

# x_train的特征矩阵（6799 * 1000）
def get_features(x):
    vec.transform(x)
    
### 贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(vec.transform(x_train),y_train)  # 特征向量与分类标签
# 查看准确率
classifier.score(vec.transform(x_test),y_test)
# 进行预测
classifier.predict(vec.transform(['This is an English sentence']))
