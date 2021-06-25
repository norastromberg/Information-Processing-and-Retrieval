#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Information Processing and Retrieval @IST 2019/2020
Course Project Part2
Exercise 3 - auxiliary classes
Group 23


"""
import os
from nltk.corpus import stopwords
import xml.etree.cElementTree as ET
from collections import OrderedDict
from nltk.util import ngrams
import ntpath
import numpy as np
from collections import Counter
from exercise1 import e1 as cscore_top_keyphrases


class DataSource:

    def __init__(self, train_dir):

        # settings
        self._train_filepaths = self.load_filepaths(train_dir)
        self._document_forest = OrderedDict()
        self._ngrammified_docs = {}

        self.parse_to_xml_forest()
        self.build_ngrammified_docs()

        return

    @property
    def ngrammified_docs(self):
        return self._ngrammified_docs

    @property
    def file_names(self):
        return [keys for keys in self._document_forest.keys()]

    @staticmethod
    def load_filepaths(tdir):
        filepaths = []
        directory = os.fsencode(tdir)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            filepaths.append(tdir + '/' + filename)

        return filepaths

    def parse_to_xml_forest(self):
        files = self._train_filepaths
        forest = OrderedDict()

        print("Parsing {} xml files".format(len(files)))

        for filepath in files[:]:
            print('.', end='', flush=True)
            path, filename = ntpath.split(filepath)
            tree = ET.parse(filepath)
            forest[filename] = tree.getroot()

        self._document_forest = forest


    @staticmethod
    def filter_out_tokens(token, stop_words, punctuation):
        word = token.find('lemma').text
        word = word.lower()

        if word in stop_words or word in punctuation:
            return True
        else:
            return False

    @staticmethod
    def __ngramify_sentence(sentence, n_range: tuple):

        ngramified_sent = []

        for i in range(*n_range):
            ngramified_sent += list(ngrams(sentence, i))

        return ngramified_sent

    @staticmethod
    def __calc_pos_weight(section, type):

        if section == 'title':
            if type == 'title':
                return 5
            else:
                return 4
        if section == 'abstract':
            if type == 'sectionHeader':
                return 3
            else:
                return 2

        return 1

    @staticmethod
    def __normalize_weights(tdict, tkey, cmin, cmax, bound_a, bound_b):

        for key, entry in tdict.items():
            entry[tkey] = (((bound_b - bound_a) * (entry[tkey] - cmin)) / (cmax - cmin)) + bound_a

        return

    def build_ngrammified_docs(self):
        forest = self._document_forest
        stop_words = stopwords.words('english')
        #stop_words.remove('of')
        # punctuation = string.punctuation
        punctuation = ['.', ',']
        ngram_range = (1, 4)
        target_field = 'lemma'  # 'word'
        max_posweight = 0

        print("\nCreating ngram docs")
        for filename, root in forest.items():
            doc_ngrams = {}
            for document in root:
                print('.', end='', flush=True)
                for sentence in document[0]:
                    new_sentence = []
                    sent_ngrams = []
                    # get pos weight
                    for token in sentence[0]:
                        if not self.filter_out_tokens(token, stop_words, punctuation):
                            new_sentence.append(token.find(target_field).text)

                        sent_ngrams = self.__ngramify_sentence(new_sentence, ngram_range)

                    # add ngrams
                    for ngram in sent_ngrams:
                        positweight = self.__calc_pos_weight(sentence.attrib['section'], sentence.attrib['type'])

                        if ngram in doc_ngrams:
                            doc_ngrams[ngram]['count'] += 1
                            doc_ngrams[ngram]['positweight'] += positweight
                            total_positweight = doc_ngrams[ngram]['positweight']
                        else:
                            doc_ngrams[ngram] = {
                                'count': 1,
                                'positweight': positweight
                            }
                            total_positweight = positweight

                        if total_positweight > max_posweight:
                            max_posweight = total_positweight

                # normalize weights
                self.__normalize_weights(doc_ngrams, 'positweight', cmin=0, cmax=max_posweight, bound_a=1, bound_b=50)

            self._ngrammified_docs[filename] = doc_ngrams

    def get_document_as_string(self, doc_name):
        forest = self._document_forest

        new_doc = ""
        root = forest[doc_name]
        for document in root:
            for sentence in document[0]:
                new_sentence = ""
                for token in sentence[0]:
                    new_sentence += (" " + token.find('lemma').text)

                new_doc += (" " + new_sentence)

        return new_doc


class KeyphraseExtractor:

    def __init__(self, datasource):
        self._datasource = datasource
        self._docs = self._datasource._ngrammified_docs
        self._features = {}
        self._default_ranking = 'fusion'
        self._rankings = {
            'tf': self.__get_top_tf,
            'idf': self.__get_top_idf,
            'tfidf': self.__get_top_tfidf,
            'bm25': self.__get_top_bm25,
            'cscore': self.__get_top_cscore,
            'rrf': self.__get_top_rrf,
            'borda': self.__get_top_borda
        }

        self.build_features()
        self.calc_score()

        return

    @property
    def ranking(self):
        return self._default_ranking

    @ranking.setter
    def ranking(self, ranking):
        self._default_ranking = ranking

    @property
    def corpus_size(self):
        return len(self._docs)

    @property
    def doc_filenames(self):
        return list(self._docs.keys())

    def build_features(self):
        docs = self._docs
        features = {}

        # unique features and docs with t
        for filename, terms in docs.items():
            for term in terms:
                if term not in features:
                    features[term] = {}
                    features[term]['count'] = 1
                else:

                    features[term]['count'] += 1

        self._features = features

    def calc_tf(self):
        docs = self._docs

        # term freq
        for filename, terms in docs.items():
            for term_attrib in terms.values():
                term_attrib['tf'] = term_attrib['count'] / len(terms)

        return

    def calc_idf(self):
        total_docs = len(self._docs)
        features = self._features

        for term, attrib in features.items():
            attrib['idf'] = np.log(total_docs / attrib['count'])

        # normalize
        cmax = max(int(attrib['idf']) for attrib in features.values())
        cmin = min(int(attrib['idf']) for attrib in features.values())
        bound_a = 0
        bound_b = 1
        #
        for term, attrib in features.items():
            attrib['idf'] = (((bound_b - bound_a) * (attrib['idf'] - cmin)) / (cmax - cmin)) + bound_a

    def calc_tfidf(self):
        docs = self._docs
        features = self._features

        self.calc_tf()
        self.calc_idf()

        for filename, terms in docs.items():
            for term, attrib in terms.items():
                attrib['tfidf'] = attrib['tf'] * features[term]['idf']

        return

    def calc_bm25(self):
        docs = self._docs
        features = self._features
        PARAM_K1 = 1.2
        PARAM_B = 0.75

        docs_total_len = sum([len(doc) for doc in docs.values()])
        avdl = docs_total_len / len(docs)

        # lenght multiplyer normalizer
        cmax = 4
        cmin = 1
        bound_a = 1
        bound_b = 10

        for filename, terms in docs.items():
            for term, attrib in terms.items():
                bm25 = features[term]['idf'] * ((attrib['tf'] * (PARAM_K1 + 1)) / (
                        attrib['tf'] + (PARAM_K1 * ((1 - PARAM_B) + PARAM_B * len(terms) / avdl))))
                # apply word count multiplier
                attrib['bm25'] = bm25 * ((((bound_b - bound_a) * (len(term) - cmin)) / (cmax - cmin)) + bound_a)

                # apply position multiplier
                attrib['bm25'] = attrib['bm25'] * attrib['positweight']

        return

    def calc_fusions(self):

        # methods to use in fusion
        methods = ('tfidf', 'bm25', 'idf')
        docs = self._docs

        for doc_name, doc in docs.items():
            for method in methods:
                top_keyphrases = self._rankings[method](doc_name, n_top=150)
                for rank, (term, score) in enumerate(top_keyphrases):

                    # Reciprocal ranking Fusion
                    if 'rrf' not in doc[term]:
                        doc[term]['rrf'] = 0

                    doc[term]['rrf'] += 1 / (50 + rank)

                    ##Sum
                    if 'borda' not in doc[term]:
                        doc[term]['borda'] = 0

                    doc[term]['borda'] += (50 - rank)

        return

    def calc_score(self):
        print("\nCalculatin Scores ")
        print('(tfifd) ', end='', flush=True)
        self.calc_tfidf()
        print('(bm25) ', end='', flush=True)
        self.calc_bm25()
        print('(rff , borda) ', end='\n')
        self.calc_fusions()

        return

    def __get_top_tf(self, doc_name, n_top=10):
        doc = self._docs[doc_name]
        terms_tfidf = Counter({term: attrib['tf'] for term, attrib in doc.items()})
        tf_rankings = [(ngram, score) for ngram, score in terms_tfidf.most_common(n_top) if score > 0]

        return tf_rankings

    def __get_top_idf(self, doc_name, n_top=10):
        doc = self._docs[doc_name]
        features = self._features

        terms_idf = Counter({term: features[term]['idf'] for term in doc.keys()})
        idf_rankings = [(ngram, score) for ngram, score in terms_idf.most_common(n_top) if score > 0]

        return idf_rankings

    def __get_top_tfidf(self, doc_name, n_top=10):
        doc = self._docs[doc_name]
        terms_tfidf = Counter({term: attrib['tfidf'] for term, attrib in doc.items()})
        tfidf_rankings = [(ngram, score) for ngram, score in terms_tfidf.most_common(n_top) if score > 0]

        return tfidf_rankings

    def __get_top_bm25(self, doc_name, n_top=10):
        doc = self._docs[doc_name]
        terms_bm25 = Counter({term: attrib['bm25'] for term, attrib in doc.items()})
        bm25_rankings = [(ngram, score) for ngram, score in terms_bm25.most_common(n_top) if score > 0]

        return bm25_rankings

    def __get_top_cscore(self, doc_name, n_top=10):
        doc = self._datasource.get_document_as_string(doc_name)

        print("\nCalculating cscore")
        terms_cscore = cscore_top_keyphrases(doc, n_top)

        cscore_rankings = []
        for term, score in terms_cscore:
            if not isinstance(term, tuple):
                term = tuple(term)
            cscore_rankings.append((term, score))

        return cscore_rankings

    def __get_top_rrf(self, doc_name, n_top=10):
        doc = self._docs[doc_name]

        terms_rff = Counter({term: (attrib['rrf'] if 'rrf' in attrib else 0) for term, attrib in doc.items()})
        rff_rankings = [(term, score) for term, score in terms_rff.most_common(n_top) if score > 0]

        return rff_rankings

    def __get_top_borda(self, doc_name, n_top=10):
        doc = self._docs[doc_name]

        terms_rff = Counter({term: (attrib['borda'] if 'borda' in attrib else 0) for term, attrib in doc.items()})
        rff_rankings = [(term, score) for term, score in terms_rff.most_common(n_top) if score > 0]

        return rff_rankings

    def get_top_keyphrases_with_scores(self, doc_name, n_top=10):
        print("\nDocname: ", doc_name)

        rankings = self._rankings[self._default_ranking](doc_name, n_top)

        print("Method:", self._default_ranking, "")

        return rankings

    def get_top_keyphrases(self, doc_name, n_top=10):

        rankings = self._rankings[self._default_ranking](doc_name, n_top)
        keyphrases = [' '.join(keys) for keys, values in rankings]

        return keyphrases
