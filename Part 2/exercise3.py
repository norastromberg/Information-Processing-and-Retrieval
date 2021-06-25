#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Information Processing and Retrieval @IST 2019/2020
Course Project Part2
Exercise 3 - An unsupervised rank aggregation approach
Group 23


"""
from exercise3_aux import DataSource
from exercise3_aux import KeyphraseExtractor
from evaluator import Evaluator


def e3():

    data = DataSource('../part1/data/train')
    extractor = KeyphraseExtractor(data)
    ev = Evaluator('data/train_combined.json', extractor)

    rankings = ('tf', 'idf', 'tfidf', 'bm25', 'borda', 'rrf')

    for ranking in rankings:
        extractor.ranking = ranking
        mean_ap = ev.compute_map()
        print("Function: {0: <7} MAP: {1}".format(ranking, mean_ap))

    return


if __name__ == '__main__':
    e3()
