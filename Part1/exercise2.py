#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Information Processing and Retrieval @IST 2019/2020
Course Project Part1
Exercise 2
Group 23

Description:

"""
import re
from xml.dom.minidom import parse
from exercise1 import SimpleApproach
import json
import os


def fetch_files(type):
    print('Fetching files ...')
    filenames = []
    if type == 'train':
        dir_string = 'data/train'
        directory = os.fsencode(dir_string)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            filenames.append('data/train/' + filename)
    elif type == 'test':
        dir_string = 'data/test_data'
        directory = os.fsencode(dir_string)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            filenames.append('data/test_data/' + filename)
    else:
        print("Need to define type of data")
    return filenames


# convert from xml-file to document
def xml_to_document(filepath):
    dom1 = parse(filepath)
    txt = dom1.getElementsByTagName('lemma')
    document = ''
    for i in range(len(txt)):
        doc = txt[i].childNodes[0].nodeValue
        re.sub(r'\W+', '', str(doc))
        document = document + ' ' + doc
    document = document.lower()
    return document


def create_corpus(xml_files):
    print('Creating corpus of ', len(xml_files), ' files')
    corpus = []
    corpus_idx = {}
    document_dict = {}
    idx = 0
    for file in xml_files:
        corpus_idx.update({file: idx})
        corpus.append(xml_to_document(file))
        document_dict[file[-8:-4]]=xml_to_document(file)
        idx = idx + 1
    return corpus, corpus_idx, document_dict


# extract candidates for a single document doc = filename
def get_candidates(sa, corpus_idx, document):
    target_id = corpus_idx.get(document)
    candidates = sa.get_top_keyphrases(target_doc_idx=target_id, n_top=20)
    return candidates



# fetch the most relevant keyphrases from the solution file
def fetch_most_relevant_keyphrases(filepath):
    with open(filepath) as f:
        keyword_dict = json.load(f)
    return keyword_dict


# extract keywords for specific document
def extract_specific_keywordlist(keyword_dict, filename):
    # funky solution for filename-extraction:
    item = filename[-8:-4]
    keywords = keyword_dict.get(item)
    result = []
    for key in keywords:
        for k in key:
            result.append(k)
    return result


# print precision,recall, and F1 scores achieved by the extraction method, per individual document
def compute_precision_recall_F1_for_all(doclist, sa, corpus_idx, key_dict):
    metrics = []
    for doc in doclist:
        keywords = extract_specific_keywordlist(key_dict, doc)
        candidates = get_candidates(sa, corpus_idx, doc)
        true_positive = []
        for phrase in candidates:
            if phrase[0] in keywords:
                true_positive.append(phrase[0])

        pr = precision(true_positive, candidates)
        re = recall(true_positive, keywords)
        f1 = F1(pr, re)
        print('\n')
        print('Metrics for document ', doc, ':')
        print('Number of true positive: ', true_positive)
        print('Document: ', doc)
        print('Precision: ', pr)
        print('Recall: ', re)
        print('F1: ', f1)
        accuracy_list = [pr, re, f1]
        metrics.append(accuracy_list)
        print('\n')
    print('List of all metrics= ',metrics, '\n')
    return metrics


# print precision,recall, and F1 scores achieved in terms of the average over the entire collection of test documents
def avg_precision_recall_f1_over_collection(metrics):
    sum_precision = 0
    sum_recall = 0
    sum_f1 = 0
    for item in metrics:
        sum_precision = sum_precision + item[0]
        sum_recall = sum_recall + item[1]
        sum_f1 = sum_f1 + item[2]
    avg_pr, avg_re, avg_f1 = (sum_precision/len(metrics), sum_recall/len(metrics), sum_f1/len(metrics))
    print('Average precision = ', avg_pr, 'Average recall = ', avg_re, 'Average f1 = ', avg_f1, '\n')
    return avg_pr, avg_re, avg_f1


# compute recall
def recall(true_positive, keywords):
    recall = len(true_positive) / len(keywords)
    return recall


# compute precision
def precision(true_positive, candidates):
    pr = len(true_positive)/len(candidates)
    return pr


# compute F1
def F1(pr, re):
    if (pr + re) <= 0:
        F1 = 0
    else:
        F1 = (2 * pr * re) / (pr + re)
    return F1


# compute precision@n, where n = rank "n"
def precision_at_n(candidates, keywords, n):
    if len(candidates)>= n:
        true_positive = 0
        for i in range(n):
            if candidates[i][0] in keywords:
                true_positive = true_positive + 1
        precision_n = true_positive/n
    return precision_n


# print mean value for the precision@5 evaluation metric
def mean_of_precision_at_5(doclist, sa, corpus_idx, key_dict):
    sum_precision_at_5 = 0
    for doc in doclist:
        keywords = extract_specific_keywordlist(key_dict,doc)
        candidates = get_candidates(sa, corpus_idx, doc)
        sum_precision_at_5 = sum_precision_at_5 + precision_at_n(candidates, keywords, 5)
    mean_of_precision_at_5 = sum_precision_at_5/len(doclist)
    print('mean_of_precision_at_5: ',mean_of_precision_at_5, '\n')
    return mean_of_precision_at_5


# compute average precision for a single document
def compute_AP(document, sa, corpus_idx, keywords):
    candidates = get_candidates(sa, corpus_idx, document)
    sum_pr_at_n= 0
    for i in range(1,len(candidates)):
        pr_at_i = precision_at_n(candidates, keywords, i)
        sum_pr_at_n = sum_pr_at_n + pr_at_i
    AP = sum_pr_at_n/len(candidates)
    return AP


# compute mean average precision
def compute_MAP(document_list,corpus, corpus_idx, key_dict):
    print('Compute mean average precision')
    sum_AP = 0
    nr = 0
    for doc in document_list:
        print('Document number ', nr, 'of ', len(document_list), ' documents')
        keywords = extract_specific_keywordlist(key_dict, doc)
        sum_AP = sum_AP + compute_AP(doc,corpus, corpus_idx, keywords)
        nr = nr + 1
    MAP = sum_AP/len(document_list)
    print('Mean average precision for the collection is : ', MAP, '\n')
    return MAP


def main():
    print("E2 (TF-IDF eval): ")

    #xml_files
    xml_files = fetch_files('train')
    print(xml_files, '\n')

    corpus, corpus_idx, doc_dict = create_corpus(xml_files)

    sa = SimpleApproach(corpus)
    sa.generate_tfidf(df_min=1, score='char')

    keyword_dict = fetch_most_relevant_keyphrases('data/train_combined.json')

    metrics = compute_precision_recall_F1_for_all(xml_files, sa, corpus_idx, keyword_dict)

    avg_precision_recall_f1_over_collection(metrics)

    mean_of_precision_at_5(xml_files, sa, corpus_idx, keyword_dict)

    compute_MAP(xml_files, sa, corpus_idx, keyword_dict)


if __name__ == "__main__":
    main()