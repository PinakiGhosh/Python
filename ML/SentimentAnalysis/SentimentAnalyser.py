#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 17:17:58 2017

@author: pinaki
"""

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer
import string

def getRootWord(word):
    syn = wordnet.synsets(word)
    rootWord=word
    if len(syn)>0:
        syn=syn[0]
        hypernyms=syn.hypernyms()
        if len(hypernyms)>0:
            lemmas=hypernyms[0].lemmas()
            if len(lemmas)>0:
                rootWord=lemmas[0].name()
    return rootWord

def create_word_features(words):
    useful_words = [getRootWord(word) for word in words if word not in stopwords.words("english")]
    useful_words = [word for word in useful_words if word not in list(string.punctuation)]
    #stemmer = SnowballStemmer("english")
    #useful_words = [stemmer.stem(word) for word in useful_words]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict

neg_reviews = []
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_reviews.append((create_word_features(words), "negative"))

pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append((create_word_features(words), "positive"))

train_set = neg_reviews[:750] + pos_reviews[:750]
test_set =  neg_reviews[750:] + pos_reviews[750:]
classifier = NaiveBayesClassifier.train(train_set)
accuracy = nltk.classify.util.accuracy(classifier, test_set)

'''
Classifying any review
review_text=""
words = word_tokenize(review_text)
words = create_word_features(words)
classifier.classify(words)
'''

syn = wordnet.synsets("verbal_creation")[0]

print(syn.hypernyms()[0].lemmas()[0].name())

type(neg_reviews[0][0])
