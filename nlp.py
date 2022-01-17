#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 15:02:44 2022

@author: YuChing
"""

import numpy as np
import pandas as pd
import matplotlib
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.datasets import load_files
import heapq
nltk.download("stopwords")


f = open('data.txt', 'r')
content = ""

#turn .txt file into string
while True:
    line = f.readline()
    
    if not line:
        break
    
    line = line.replace('\n', ". ") 
    content += line

f.close()

# --- A.	Show the most frequency noun mentioned by CEO. ---
words = word_tokenize(content) 

words_POS = nltk.pos_tag(words)
dicNNP = {}
dicNN = {}
for noun, prop in words_POS:
    if prop == 'NNP':
        if noun in dicNNP:
            dicNNP[noun] += 1
        else:
            dicNNP[noun] = 1
            
    if prop == 'NN':
        if noun in dicNN:
            dicNN[noun] += 1
        else:
            dicNN[noun] = 1


sorted_dicNNP = sorted(dicNNP.items(), key=lambda item: item[1], reverse=True)
sorted_dicNN = sorted(dicNN.items(), key=lambda item: item[1], reverse=True)
arrNNP = []
arrNNPTime = []
arrNN = []
arrNNTime = []

count = 0

for noun, times in sorted_dicNNP:
    arrNNP.append(noun)
    arrNNPTime.append(times)
    count += 1
    
    if count == 6:
        break
  
count = 0

for noun, times in sorted_dicNN:
    arrNN.append(noun)
    arrNNTime.append(times)
    count += 1
    
    if count == 8:
        break
    
left = np.array(arrNNP)
height = np.array(arrNNPTime)
matplotlib.pyplot.bar(left, height)
matplotlib.pyplot.show()

left = np.array(arrNN)
height = np.array(arrNNTime)
matplotlib.pyplot.bar(left, height)
matplotlib.pyplot.show()

# ---End of part A.	Show the most frequency noun mentioned by CEO. ---
    
# --- C. Condense the speech into a summary ---

clean_content = content.lower()
clean_content = re.sub(r'\W', ' ', clean_content)
clean_content = re.sub(r'\d', ' ', clean_content)
clean_content = re.sub(r'\s+', ' ', clean_content)
clean_words = word_tokenize(clean_content)


sentences = sent_tokenize(content)
stop_words = nltk.corpus.stopwords.words('english')

word2count = {}
for word in clean_words:
    if word not in stop_words:
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
            
for key in word2count:
    word2count[key] = word2count[key]/max(word2count.values())
    
sent2score = {}
for sentence in sentences:
    for word in nltk.word_tokenize(sentence.lower()):
        if word in word2count:
            if len(sentence.split(' ')) < 30:
                if sentence not in sent2score:
                    sent2score[sentence] = word2count[word]
                else:
                    sent2score[sentence] += word2count[word]
                
best_sentences = heapq.nlargest(10, sent2score, key = sent2score.get)
summary = ""
for sent in best_sentences:
    summary += sent
    
print("Summary:")
print(summary)
print("-------------------------------------------------")

# --- End of C. Condense the speech into a summary ---
    

# --- B. Calculate the sentiment of each sentences. ---
    
with open('tfidfmodel.pickle', 'rb') as f:
    vectorizer = pickle.load(f)
    
with open('classifier.pickle', 'rb') as f:
    clf = pickle.load(f)
    
count = 0
total = 0
year = {}

for sentence in sentences:
    words = sentence.split(" ")
    for word in words:
        if word.isdigit() and int(word)>2000:
            if word in year:
                year[word].append(sentence)
            else:
                year[word] = [sentence]
            break
        
    oriSent = sentence
    sentence = sentence.lower()
    sentence = re.sub(r"that's","that is",sentence)
    sentence = re.sub(r"there's","there is",sentence)
    sentence = re.sub(r"what's","what is",sentence)
    sentence = re.sub(r"where's","where is",sentence)
    sentence = re.sub(r"it's","it is",sentence)
    sentence = re.sub(r"who's","who is",sentence)
    sentence = re.sub(r"i'm","i am",sentence)
    sentence = re.sub(r"she's","she is",sentence)
    sentence = re.sub(r"he's","he is",sentence)
    sentence = re.sub(r"they're","they are",sentence)
    sentence = re.sub(r"who're","who are",sentence)
    sentence = re.sub(r"ain't","am not",sentence)
    sentence = re.sub(r"wouldn't","would not",sentence)
    sentence = re.sub(r"shouldn't","should not",sentence)
    sentence = re.sub(r"can't","can not",sentence)
    sentence = re.sub(r"couldn't","could not",sentence)
    sentence = re.sub(r"won't","will not",sentence)
    sentence = re.sub(r"we're","we are",sentence)
    sentence = re.sub(r"don't","do not",sentence)
    sentence = re.sub(r"we've","we have",sentence)
    sentence = re.sub(r"we'll","we will",sentence)
    sentence = re.sub(r"\W"," ",sentence)
    sentence = re.sub(r"\d"," ",sentence)
    sent = clf.predict(vectorizer.transform([sentence]))
    if sent == 1:
        count += 1
    total += 1
    


df = pd.DataFrame([
    ["Positive", count], ["Nagative", total - count]], columns = ["Senti", "num"])
    
matplotlib.pyplot.pie(df["num"], labels=df["Senti"], autopct="%1.2f%%")
matplotlib.pyplot.title("CEO Sentiment Analysis")
matplotlib.pyplot.show()

# --- End of B. Calculate the sentiment of each sentences. ---

print("Financial year sentences:")
print(year)

    
