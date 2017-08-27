# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 09:14:37 2017
@author: Junior
新闻主题分类(多分类问题)-朴素贝叶斯

知识关键点：
- 2:8划分测试与训练数据集
- 函数用法：zip(),sorted(key=),.isdigit()
- 构造0,1特征向量(取词频TOP1000的关键词)
- sklearn中的多项式分类_MultinomialNB().fit
"""

import os
os.chdir(r'F:\付费学习\破解\机器学习全套课程\codec_myself\08')
os.getcwd()

import codecs
import random
import jieba
# import sklearn
from sklearn.naive_bayes import MultinomialNB
# import numpy as np
# import pylab as pl
import matplotlib.pyplot as plt


### 停用词统计
def make_word_set(words_file):
    words_set = set()
    with codecs.open(words_file,'r','utf-8') as fp:
        for line in fp:
            word = line.strip()
            if len(word) > 0 and word not in words_set:
                words_set.add(word)
    return words_set


### 文本处理
def text_processing(folder_path,test_size=0.2):
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []
    ### way1 遍历文件夹
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path,folder)
        files = os.listdir(new_folder_path)
        j = 1
        for file in files:
            if j > 100: # 防止内存爆掉
                break
            with codecs.open(os.path.join(new_folder_path,file),'r','utf-8') as fp:
                raw = fp.read()
                word_cut = jieba.cut(raw,cut_all = False)
                word_list = list(word_cut)
                data_list.append(word_list) # 训练集
                class_list.append(folder)   # 类别
                j += 1      
    ### way2
    # filePaths = [];
    # fileContents = [];
    # for root, dirs, files in os.walk(folder_path):
    #     for name in files:
    #         filePath = os.path.join(root, name);
    #         filePaths.append(filePath);
    #         f = codecs.open(filePath, 'r', 'utf-8')
    #         fileContent = f.read()
    #         fileContent_cut = jieba.cut(fileContent,cut_all = False)
    #         fileContent_list = list(fileContent_cut)
    #         f.close()
    #         fileContents.append(fileContent_list)

    # 划分训练集和测试集
    data_class_list = list(zip(data_list, class_list))
    random.shuffle(data_class_list)    # 打乱顺序
    index = int(len(data_class_list) * test_size) + 1   # 抽取测试数据集的占比
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list,train_class_list = zip(*train_list) # 特征与标签
    test_data_list,test_class_list = zip(*test_list) 
    # 可以用sklearn完成
    # from sklearn.cross_validation import train_test_split
    # train_data_list, test_data_list, train_class_list, test_class_list = sklearn.cross_validation.train_test_split(data_list, class_list, test_size=test_size)

    # 统计词频
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict:
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
            
    # 降序排序（key函数）
    all_words_tuple_list = sorted(all_words_dict.items(),key = lambda f:f[1], reverse = True)
    all_words_list = list(zip(*all_words_tuple_list))[0]
    
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


### 选取前1000个特征词
def words_dict(all_words_list,deleteN,stopwords_set=set()):
    feature_words = []
    n = 1
    for t in range(deleteN,len(all_words_list),1):
        if n > 1000:    # 最多取1000个维度
            break
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
            n += 1
    return feature_words
            

### 文本特征
def text_features(train_data_list, test_data_list, feature_words):
    def text_features(text,feature_words):
        # text = train_data_list[0]
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
        
    # 0,1的矩阵（1000列-维度）    
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list,test_feature_list
    

### 分类
def text_classifier(train_feature_list,test_feature_list,train_class_list,test_class_list):
    # sklearn多项式分类器
    classifier = MultinomialNB().fit(train_feature_list,train_class_list)   # 特征向量与类别
    test_accuracy = classifier.score(test_feature_list,test_class_list)
    return test_accuracy
    
    

if __name__ == '__main__':
    # 文本预处理（分词、划分训练与测试集、排序）
    folder_path = r'.\Lecture_2\Naive-Bayes-Text-Classifier\Database\SogouC\Sample'
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = text_processing(folder_path, test_size=0.2)
    
    # stopwords_set
    stopwords_file = r'.\Lecture_2\Naive-Bayes-Text-Classifier\stopwords_cn.txt'
    stopwords_set = make_word_set(stopwords_file)

    # 特征提取与分类
    flag = 'sklearn'
    deleteNs = range(0,1000,20)
    test_accuracy_list = []
    for deleteN in deleteNs:
        # 前1000个特征词
        feature_words = words_dict(all_words_list,deleteN,stopwords_set)
        # 计算特征向量
        train_feature_list, test_feature_list = text_features(train_data_list,test_data_list,feature_words)
        # sklearn分类器计算准确度
        test_accuracy = text_classifier(train_feature_list,test_feature_list,train_class_list,test_class_list)
        # 不同特征向量下的准确度
        test_accuracy_list.append(test_accuracy)
    print(test_accuracy_list)
    
    # 结果评价
    plt.figure()
    plt.plot(deleteNs,test_accuracy_list)
    plt.title('Relationship of deleteNs and test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.savefig('result.png',dpi = 100)
    plt.show()

