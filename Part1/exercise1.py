#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Information Processing and Retrieval @IST 2019/2020
Course Project Part1
Exercise 1
Group 23

Description: Exercise 1 of the project. The function get_top_keyphrases() returns the top keyphases of a target document extracted with a simple aproach

"""

import sys
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import re


def file_and_20newsgroups(file_path):
    try:
        f = open(file_path, 'r')
        document = f.read()
        f.close()
    except Exception as e:
        print(e)
        exit()

    categories = ['comp.graphics', 'sci.electronics', 'comp.sys.ibm.pc.hardware', 'sci.space']
    doc_set = [document]
    doc_set += fetch_20newsgroups(subset='train', categories=categories).data[:100]

    return doc_set


class SimpleApproach:

    def __init__(self, documents):
        self._documents = documents
        self.__generate_corpus()

    def __generate_corpus(self):
        # Remove stand alone numbers and  underscores the rest is done by vectorizer
        self._corpus = [re.sub(r'[0-9_]+', '', document) for document in self._documents]

    def generate_tfidf(self, df_min=1, score='char'):
        vectorizer = TfidfVectorizer(encoding='utf-8', decode_error='strict', strip_accents='unicode', lowercase=True,
                                     analyzer='word', stop_words='english', ngram_range=(1, 3), max_df=1.0,
                                     min_df=df_min,
                                     use_idf=True, smooth_idf=True, norm='l2')

        csr_matrix = vectorizer.fit_transform(self._corpus)
        tfidf_matrix = csr_matrix.todense()
        features = vectorizer.get_feature_names()

        # Multiply factors can be character length or word count
        score = (lambda x: len(re.findall(r'\w+', x))) if score == 'word' else (lambda x: len(x))
        multiply_factors = [score(feature) for feature in features]
        tfidf_matrix = np.multiply(tfidf_matrix, multiply_factors)

        self._features = features
        self._tfidf_matrix = tfidf_matrix


    def get_top_keyphrases(self, target_doc_idx=0, n_top=5):

        # Get target document results
        values = self._tfidf_matrix.tolist()[target_doc_idx]
        rankings = Counter(dict(zip(self._features, values)))
        top_keyphrases = [(x,y) for x, y in rankings.most_common(n_top) if y > 0]

        return top_keyphrases


def e1():
    if len(sys.argv) < 2:
        file_path = 'data/exercise-1-text.txt'
    else:
        file_path = sys.argv[1]

    documents = file_and_20newsgroups(file_path)

    sa = SimpleApproach(documents)
    sa.generate_tfidf(df_min=1, score='word')

    keyphrases = sa.get_top_keyphrases(target_doc_idx=0, n_top=5)

    print("E1 (TF-IDF) simple aproach: ")
    print("file: " + file_path + "\nKeyphrases: ")
    print(keyphrases)


if __name__ == '__main__':
    e1()
