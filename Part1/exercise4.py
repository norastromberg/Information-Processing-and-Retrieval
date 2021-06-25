#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Information Processing and Retrieval @IST 2019/2020
Course Project Part1
Exercise 4
Group 23

Description:


"""
import nltk
import math
import numpy as np
import re
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from exercise2 import *
from exercise3 import bm25_get_top_keyphrases

#The documents are a list of all the documents where each document is a list
def read_files(files):
    dict = {}
    documents = []
    for file in files:
        documents.append(xml_to_document(file))
        dict[file[-8:-4]] = xml_to_document(file)
    return documents,dict

#each documents keyphrases is a list in the list of keyphrases
def get_keyphrases(file, document_paths):
    dict = fetch_most_relevant_keyphrases(file)
    keyphrase_dict = {}
    for doc in document_paths:
        keyphrases = extract_specific_keywordlist(dict,doc)
        keyphrase_dict[doc[-8:-4]] = keyphrases
    return keyphrase_dict


def make_sentences(document):
    return nltk.sent_tokenize(document)


def make_word_list(document):
    sentences = make_sentences(document)
    words = []
    for sentence in sentences:
        word_list = nltk.word_tokenize(sentence)
        for w in word_list:
            words.append(w)
    return words


#Words that appear early in the document should be more weighted than those that appear later.
def term_position(document,candidate):
    sentences = make_sentences(document)
    pos = 0
    for i,sentence in enumerate(sentences):
        if (candidate in sentence and pos == 0):
            pos = i + 1
    if pos == 0:
        return 0
    return 1/math.sqrt(pos)


def term_frequency(document,candidate):
    return len(re.findall(candidate,str(document)))

def inverse_doc_freq(documents,document,candidate):
    df = term_frequency(document,candidate)
    number_of_docs_where_term_appear = 0
    for i,doc in enumerate(documents):
        if term_frequency(doc,candidate)!= 0:
            number_of_docs_where_term_appear +=1
    if number_of_docs_where_term_appear == 0:
        return 0
    return math.log(len(documents) /number_of_docs_where_term_appear)


def convert_from_candidate_to_attributes_simple(document, candidate):
    return term_position(document,candidate), len(candidate), term_frequency(document,candidate)


#When we add values to this one we have to change the number of attributes in make_x_y, according to the number of attributes
def convert_from_candidate_to_attributes_advanced(documents,document, candidate, tfidf_or_bm25):
    # return term_position(document,candidate), len(candidate), term_frequency(document,candidate),\
    #        inverse_doc_freq(documents,document,candidate),tfidf_or_bm25
    return term_position(document, candidate), \
           inverse_doc_freq(documents, document, candidate), tfidf_or_bm25


def get_class(candidate, real_candidates):
    if candidate in real_candidates:
        return 1
    else :
        return 0


#5 is the number of candidates returned from get_top_keyphrases, can be changed
#doc_id_dict = {filepath : index of document}
#Filename = filepath[11:15]
#file_doc_dict = {filename : document}

def make_x_y(documents,doc_id_dict,file_doc_dict, keyphrases_dict):
    number_of_candidates = 5*len(documents)
    number_of_attributes = 3
    x = np.ndarray(shape = (number_of_candidates,number_of_attributes))
    y = np.ndarray(shape = number_of_candidates)
    count = 0
    for i,doc_path in enumerate(doc_id_dict.keys()):
        doc_name = doc_path[-8:-4]
        document = file_doc_dict[doc_name]
        document_index = doc_id_dict[doc_path]
        candidates = bm25_get_top_keyphrases(documents,target_doc_idx=document_index)
        for candidate in candidates:
            #x[count:] = convert_from_candidate_to_attributes_simple(document,candidate)
            #Tfidf value if get_top_keyphrase is used, bm25 if bm25_get_top_keyphrase is used
            tfidf_or_bm25 = candidate[1]
            x[count:] = convert_from_candidate_to_attributes_advanced(documents,document,candidate[0],tfidf_or_bm25)
            y[count] = get_class(candidate[0], keyphrases_dict[doc_name])
            count += 1
    return x, y

def naive_bayes(x_train, y_train):
    clf = GaussianNB()
    clf.fit(x_train,y_train)
    return clf

def perceptron(x_train,y_train):
    clf = Perceptron( random_state=0)
    clf.fit(x_train, y_train)
    return clf

def test_perceptron(x_test, y_test, clf):
    print("predicted values : " + str(clf.predict(x_test)))
    return clf.score(x_test, y_test)

def find_best_candidates(document,candidates, clf):
    x = np.ndarray(shape=(len(candidates),3))
    good_candidates = []
    for i,candidate in enumerate(candidates):
        x[i:] = convert_from_candidate_to_attributes_simple(document,candidate)
    predicted = clf.predict(x)
    for i,n in enumerate(predicted):
        if n == 1:
            good_candidates.append(candidate[i])
    return good_candidates

def main():
    print("E4 (Supervised Approach ): ")

    training_files = fetch_files("train")
    training_documents, doc_id_dict, file_doc_dict = create_corpus(training_files)

    training_keyphrases = get_keyphrases('data/train_combined.json',training_files)
    x_train,y_train = make_x_y(training_documents,doc_id_dict,file_doc_dict,training_keyphrases)
    print(x_train,y_train)
    clf = perceptron(x_train, y_train)
    #clf = naive_bayes(x_train, y_train)

    test_files = fetch_files("test")
    test_documents, doc_id_dict, file_doc_dict = create_corpus(test_files)
    test_keyphrases = get_keyphrases('data/test.combined.stem.json',test_files)
    x_test, y_test = make_x_y(test_documents,doc_id_dict,file_doc_dict,test_keyphrases)
    print(x_test,y_test)

    test_perceptron(x_test,y_test)
    acc = test_perceptron(x_test,y_test,clf)
    print("accuracy: " + str(acc))

if __name__ == "__main__":
    main()
