#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Information Processing and Retrieval @IST 2019/2020
Course Project Part2
Evaluator generalized use
Group 23

Description:

"""
import json
import os


class Evaluator:

    def __init__(self, keyphrase_file, extractor):
        self._best_keyphrases = self.__parse_best_keyphrases(keyphrase_file)
        self._extractor = extractor
        self._ap = {}

    @staticmethod
    def __parse_best_keyphrases(keyphrase_file):
        with open(keyphrase_file) as f:
            json_dict = json.load(f)

        # flatten lists
        best_keyphrases = {}
        for doc, list_of_lists in json_dict.items():
            flat_list = []
            for sublist in list_of_lists:
                for item in sublist:
                    flat_list.append(item)

            best_keyphrases[doc] = flat_list
        return best_keyphrases

    @staticmethod
    def __precision(retrieved, relevant):
        common = set(retrieved) & set(relevant)
        precision = len(common) / len(retrieved)

        return precision

    def precision_at_k(self, retrieved, relevant, k):

        return self.__precision(retrieved[:k], relevant[:k])

    def compute_ap(self, document_name):

        extractor = self._extractor
        filename, file_extension = os.path.splitext(document_name)
        relevant = self._best_keyphrases[filename]
        retrieved = extractor.get_top_keyphrases(document_name, n_top=10)

        sum = 0
        for k, term in enumerate(retrieved):
            sum += self.precision_at_k(retrieved, relevant, k + 1) * (1 if (term in relevant) else 0)

        ap = sum / len(relevant)

        return ap

    def compute_map(self):

        extractor = self._extractor
        corpus_size = extractor.corpus_size
        doc_filenames = extractor.doc_filenames

        sum = 0

        for doc in doc_filenames:

            sum += self.compute_ap(doc)

        map = sum / corpus_size

        return map
