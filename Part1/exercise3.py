#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Information Processing and Retrieval @IST 2019/2020
Course Project Part1
Exercise 3
Group 23

Description:

Exercise 3.1 : Limit candidates to only noun candidates
Input : list of all candidates
Output : list of noun candidates
"""
import sys
import nltk
from nltk.corpus import stopwords
import string
from nltk.util import ngrams
import numpy as np
from collections import Counter
from exercise1 import file_and_20newsgroups
import re
from evaluator import Evaluator

from exercise2 import fetch_files, fetch_most_relevant_keyphrases, create_corpus


def bm25_get_top_keyphrases(documents, df_min=1, target_doc_idx=0, n_top=5, term_len=False):
    ia = ImprovedApproach(documents)
    ia.generate_bm25(df_min, term_len)
    top_keyphrases = ia.get_top_keyphrases(target_doc_idx, n_top)

    return top_keyphrases


class ImprovedApproach:

    def __init__(self, documents):
        self._documents = documents
        self._corpus = None
        self.__generate_corpus()
        self._bm25_matrix = None
        self._features = None

    def __generate_corpus(self):
        self._corpus = [self.__prepare_document(document) for document in self._documents]

    def get_top_keyphrases(self, target_doc_idx=0, n_top=5):

        matrix = self._bm25_matrix
        features = self._features
        values = matrix.tolist()[target_doc_idx]
        terms = [" ".join(ngram) for ngram in features]
        rankings = Counter(dict(zip(terms, values)))
        return [(x, y) for x, y in rankings.most_common(n_top) if y > 0]

    def generate_bm25(self, df_min=1, term_len=False):

        n_docs = len(self._documents)
        n_docs_with_t = 0
        t_freq: float
        avdl: float
        n_terms_in_doc = np.zeros(n_docs)
        features = []
        PARAM_K1 = 1.2
        PARAM_B = 0.75

        corpus = self._corpus

        # Calc number of termns in doc and get unique features
        for idx, doc in enumerate(corpus):
            for term in doc:
                n_terms_in_doc[idx] += 1
                if term not in features:
                    features.append(term)

        avdl = sum(n_terms_in_doc) / n_docs
        matrix = np.zeros((n_docs, len(features)))

        # Populate with term freq
        for row, doc in enumerate(corpus):
            for column, term in enumerate(features):
                matrix[row, column] = doc.count(term) / n_terms_in_doc[row] if n_terms_in_doc[row] > 0 else 0

        # calculate idf
        idf = np.zeros(len(features))
        for column, term in enumerate(features):
            n_docs_with_t = np.count_nonzero(matrix[:, column])
            idf[column] = np.log((n_docs - n_docs_with_t + 0.5) / (n_docs_with_t + 0.5))

        # normalize idf
        idf = self.__normalise_array(idf)

        # Apply Formula
        for row, doc in enumerate(corpus):
            for column, term in enumerate(features):
                t_freq = matrix[row, column]
                matrix[row, column] = idf[column] * ((t_freq * (PARAM_K1 + 1)) / (
                        t_freq + (PARAM_K1 * ((1 - PARAM_B) + PARAM_B * n_terms_in_doc[row] / avdl))))

        # Improvements

        # Apply weights for longer terms
        if term_len:
            multiply_factors = [len(" ".join(feature)) for feature in features]
            matrix = np.multiply(matrix, multiply_factors)

        self._bm25_matrix = matrix
        self._features = features

    @staticmethod
    def __normalise_array(vec):
        vec -= vec.min()
        vec /= vec.ptp()
        vec += 1
        return vec

    @staticmethod
    def __part_of_speech_tag(candidates):
        tagged_candidates = nltk.pos_tag(candidates)
        new_candidates = []
        # print(tagged_candidates)
        for candidate in tagged_candidates:
            if (candidate[1].startswith("NN")):
                new_candidates.append(candidate[0])
        return new_candidates

    @staticmethod
    def __ngramify(sentences, n_range: tuple):
        """Returns list of ngrams

            Parameters
            ----------
            sentences : list of list of str
                A list of sentences belonging to a Document
            ngram_range : tuple
                range of ngrams to create

            Returns
            ----------
                a list of tuples ngrams
        """
        ngramified_doc = []

        for i in range(*n_range):
            for sentence in sentences:
                ngramified_doc += list(ngrams(sentence, i))

        return ngramified_doc

    def __prepare_document(self, document):
        """Returns list of documents terms as tuples

            Parameters
            ----------
            document : str
                a document as a string

            Returns
            ----------
                a list of tuples ngrams
        """

        stop_words = stopwords.words('english')
        punctuation = string.punctuation
        ngram_range = (1, 4)

        # Lower Case
        document = document.lower()
        # Tokenize sentence
        sentences = nltk.sent_tokenize(document)
        # Tokenize words
        sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
        # remove stop words
        sentences = [[word for word in sentence if word not in stop_words] for sentence in sentences]
        # Select only words (no punctuation)
        sentences = [[re.sub(r'[^\w\s]', '', word) for word in sentence] for sentence in sentences]
        # Remove numbers
        sentences = [[re.sub(r'[0-9_]+', '', word) for word in sentence] for sentence in sentences]
        # Remove empty words
        sentences = [[word for word in sentence if word is not ""] for sentence in sentences]
        # Select parts of speech
        sentences = [self.__part_of_speech_tag(sentence) for sentence in sentences]
        # Gramify document
        doc_ngrams = self.__ngramify(sentences, ngram_range)

        return doc_ngrams


def evaluate_approach():
    e3_evaluator = Evaluator(documents_type='train')
    extractor = ImprovedApproach(e3_evaluator.corpus)
    extractor.generate_bm25(df_min=1, term_len=True)
    e3_evaluator.extractor = extractor

    metrics = e3_evaluator.compute_precision_recall_F1_for_all()
    e3_evaluator.avg_precision_recall_f1_over_collection(metrics)
    e3_evaluator.mean_of_precision_at_5()
    e3_evaluator.compute_MAP()


def e3():
    if len(sys.argv) < 2:
        file_path = 'data/exercise-1-text.txt'
    else:
        file_path = sys.argv[1]

    documents = file_and_20newsgroups(file_path)

    keyphrases = bm25_get_top_keyphrases(documents, target_doc_idx=0, n_top=5)

    print("E3 (TF-IDF improved): ")
    print("file: " + file_path + "\nKeyphrases: ")
    print(keyphrases)


if __name__ == "__main__":
    #e3()
    evaluate_approach()
