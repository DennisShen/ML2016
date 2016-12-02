#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 15:21:05 2016

@author: dennis
"""

import sys
import numpy as np
import pandas as pd
import keras.preprocessing.text
from   sklearn.cluster import KMeans
import nltk
from   nltk.corpus import stopwords
from   nltk.tokenize import RegexpTokenizer
import random

""" stop word and tokenize """
nltk.download("stopwords")
stop = set(stopwords.words('english'))
stop.add(u'using')
tokenizer = RegexpTokenizer(r'\w+')


""" tfidf """
doc = []
with open(sys.argv[1] + 'title_StackOverflow.txt', 'r') as textfile:  
    for line in textfile:
        words = tokenizer.tokenize(line.lower())
        filtered_words = [w for w in words if not w in stop]
        filtered_line = ' '.join(filtered_words)
        if len(filtered_line):
            doc.append(filtered_line)    
             
tk = keras.preprocessing.text.Tokenizer(nb_words=20, split=" ")
tk.fit_on_texts(doc)
tk.fit_on_sequences(doc)
tfidf = tk.texts_to_matrix(doc, mode="tfidf")

sum_of_vector = tfidf.sum(axis=1)

""" k-mean for list with top 20 keyword """
""" random for list without keyword """
list_with_keyword    = [w for w, s in zip(tfidf, sum_of_vector) if s != 0]
list_without_keyword = [w for w, s in zip(tfidf, sum_of_vector) if s == 0]
                        
id_with_keyword    = [idx for idx, s in enumerate(sum_of_vector) if s != 0]
id_without_keyword = [idx for idx, s in enumerate(sum_of_vector) if s == 0]
                      
km = KMeans(n_clusters=18, init='k-means++', max_iter=100, n_init=1)  
km.fit(list_with_keyword)

ans_with_keyword = np.array(km.labels_).astype('int')
ans_without_keyword = np.random.randint(0, 18, size=(len(list_without_keyword),)).astype('int')

""" combine two list """
ans_list = []
count_with_keyword = 0
count_without_keyword = 0
for id in range(len(doc)):
    if id == id_with_keyword[count_with_keyword]:
        ans_list.append(ans_with_keyword[count_with_keyword])
        count_with_keyword = count_with_keyword + 1
    else:
        ans_list.append(ans_without_keyword[count_without_keyword])
        count_without_keyword = count_without_keyword + 1
        
""" write out prediction """
df = pd.read_csv(sys.argv[1] + 'check_index.csv', header=None, dtype={"ID": int, "x_ID": int, "y_ID": int})   

df.drop(0, axis=1, inplace=True)
df.drop(0, axis=0, inplace=True)

check_x = np.array(df.values[:,0])
check_y = np.array(df.values[:,1])

x = check_x.astype(int)
y = check_y.astype(int)

del df
del check_x
del check_y

with open(sys.argv[2], 'w') as f:
    f.write('ID,Ans\n')
    for idx in range(5000000):
        if ans_list[x[idx]] == ans_list[y[idx]]:
            f.write('%d,%d\n'%(idx, 1))
        else:
            f.write('%d,%d\n'%(idx, 0))
