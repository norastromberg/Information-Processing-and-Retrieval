#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Information Processing and Retrieval @IST 2019/2020
Course Project Part1
Evaluator generalized use
Group 23

Description:

"""
import re
from xml.dom.minidom import parse
from exercise1 import SimpleApproach
import json
import os


class Evaluator:

    def __init__(self, documents_type):
        self._xml_files = self.__fetch_files(documents_type)[0:10]
        print(self._xml_files, '\n')
        self._extractor = None
        self._keyword_dict = self.__fetch_most_relevant_keyphrases('data/train_combined.json')

        self._corpus, self._corpus_idx, self._doc_dict = self.__create_corpus()

    @property
    def corpus(self):
        return self._corpus

    @property
    def extractor(self):
        return self._extractor

    @extractor.setter
    def extractor(self, ext):
        self._extractor = ext

    @staticmethod
    def __fetch_files(type):
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
    @staticmethod
    def __xml_to_document(filepath):
        dom1 = parse(filepath)
        txt = dom1.getElementsByTagName('lemma')
        document = ''
        for i in range(len(txt)):
            doc = txt[i].childNodes[0].nodeValue
            re.sub(r'\W+', '', str(doc))
            document = document + ' ' + doc
        document = document.lower()
        return document


    def __create_corpus(self):

        xml_files = self._xml_files
        print('Creating corpus of ', len(xml_files), ' files')
        corpus = []
        corpus_idx = {}
        document_dict = {}
        idx = 0
        for file in xml_files:
            corpus_idx.update({file: idx})
            corpus.append(self.__xml_to_document(file))
            document_dict[file[-8:-4]] = self.__xml_to_document(file)
            idx = idx + 1
        return corpus, corpus_idx, document_dict

    # extract candidates for a single document doc = filename
    def __get_candidates(self, document):
        corpus_idx = self._corpus_idx
        extractor = self._extractor

        target_id = corpus_idx.get(document)
        candidates = extractor.get_top_keyphrases(target_doc_idx=target_id, n_top=20)
        return candidates

    # fetch the most relevant keyphrases from the solution file
    @staticmethod
    def __fetch_most_relevant_keyphrases(filepath):
        with open(filepath) as f:
            keyword_dict = json.load(f)
        return keyword_dict

    # extract keywords for specific document
    def __extract_specific_keywordlist(self, filename):
        # funky solution for filename-extraction:
        item = filename[-8:-4]
        keywords = self._keyword_dict.get(item)
        result = []
        for key in keywords:
            for k in key:
                result.append(k)
        return result

    # print precision,recall, and F1 scores achieved by the extraction method, per individual document
    def compute_precision_recall_F1_for_all(self):
        metrics = []
        doclist = self._xml_files
        key_dict = self._keyword_dict

        for doc in doclist:
            keywords = self.__extract_specific_keywordlist(doc)
            candidates = self.__get_candidates(doc)
            true_positive = []
            for phrase in candidates:
                if phrase[0] in keywords:
                    true_positive.append(phrase[0])

            pr = self.__precision(true_positive, candidates)
            re = self.__recall(true_positive, keywords)
            f1 = self.__F1(pr, re)
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
        print('List of all metrics= ', metrics, '\n')
        return metrics

    # print precision,recall, and F1 scores achieved in terms of the average over the entire collection of test documents
    def avg_precision_recall_f1_over_collection(self, metrics):
        sum_precision = 0
        sum_recall = 0
        sum_f1 = 0
        for item in metrics:
            sum_precision = sum_precision + item[0]
            sum_recall = sum_recall + item[1]
            sum_f1 = sum_f1 + item[2]
        avg_pr, avg_re, avg_f1 = (sum_precision / len(metrics), sum_recall / len(metrics), sum_f1 / len(metrics))
        print('Average precision = ', avg_pr, 'Average recall = ', avg_re, 'Average f1 = ', avg_f1, '\n')
        return avg_pr, avg_re, avg_f1

    # compute recall
    @staticmethod
    def __recall(true_positive, keywords):
        recall = len(true_positive) / len(keywords)
        return recall

    # compute precision
    @staticmethod
    def __precision(true_positive, candidates):
        pr = len(true_positive) / len(candidates)
        return pr

    # compute F1
    @staticmethod
    def __F1(pr, re):
        if (pr + re) <= 0:
            F1 = 0
        else:
            F1 = (2 * pr * re) / (pr + re)
        return F1

    # compute precision@n, where n = rank "n"
    @staticmethod
    def __precision_at_n(candidates, keywords, n):
        if len(candidates) >= n:
            true_positive = 0
            for i in range(n):
                if candidates[i][0] in keywords:
                    true_positive = true_positive + 1
            precision_n = true_positive / n
        return precision_n

    # print mean value for the precision@5 evaluation metric
    def mean_of_precision_at_5(self):
        doclist = self._xml_files
        sum_precision_at_5 = 0
        for doc in doclist:
            keywords = self.__extract_specific_keywordlist(doc)
            candidates = self.__get_candidates(doc)
            sum_precision_at_5 = sum_precision_at_5 + self.__precision_at_n(candidates, keywords, 5)
        mean_of_precision_at_5 = sum_precision_at_5 / len(doclist)
        print('mean_of_precision_at_5: ', mean_of_precision_at_5, '\n')
        return mean_of_precision_at_5

    # compute average precision for a single document
    def __compute_AP(self, document, keywords):
        candidates = self.__get_candidates(document)
        sum_pr_at_n = 0
        for i in range(1, len(candidates)):
            pr_at_i = self.__precision_at_n(candidates, keywords, i)
            sum_pr_at_n = sum_pr_at_n + pr_at_i
        AP = sum_pr_at_n / len(candidates)
        return AP

    # compute mean average precision
    def compute_MAP(self):
        document_list = self._xml_files
        print('Compute mean average precision')
        sum_AP = 0
        nr = 0
        for doc in document_list:
            print('Document number ', nr, 'of ', len(document_list), ' documents')
            keywords = self.__extract_specific_keywordlist(doc)
            sum_AP = sum_AP + self.__compute_AP(doc, keywords)
            nr = nr + 1
        MAP = sum_AP / len(document_list)
        print('Mean average precision for the collection is : ', MAP, '\n')
        return MAP


def main():
    print("E2 (TF-IDF eval): ")

    # xml_files
    #xml_files = fetch_files('train')[0:10]
    #print(xml_files, '\n')

    e2_evaluator = Evaluator(documents_type='train')

    sa = SimpleApproach(e2_evaluator.corpus)
    sa.generate_tfidf(df_min=1, score='char')
    e2_evaluator.extractor = sa

    metrics = e2_evaluator.compute_precision_recall_F1_for_all()

    e2_evaluator.avg_precision_recall_f1_over_collection(metrics)

    e2_evaluator.mean_of_precision_at_5()

    e2_evaluator.compute_MAP()


if __name__ == "__main__":
    main()
