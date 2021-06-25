#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Information Processing and Retrieval @IST 2019/2020
Course Project Part2
Exercise 1 - An approach based on graph ranking
Group 23


"""

"""
In this first exercise, you should develop a program that uses a PageRank-based method for
extracting keyphrases, taking the following considerations into account:

-   You should start by creating a graph where nodes correspond to n-grams (where 1 ≤
    n ≤ 3, and ignoring stop-words and punctuation) from the document being processed, and
    where edges encode co-occurrences within the same sentence;
-
"""
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


"""TODO : The document should come from a document on the harddrive, need to implement method for catching this document.
          Also, have to find the correct alpha value for pagerank"""

#candidates and links have to be lists
def make_graph(candidates,links):
    G = nx.Graph()
    G.add_nodes_from(candidates)
    G.add_edges_from(links)
    return G

def visualize_graph(graph):
    nx.draw(graph,with_labels=True)
    plt.show()

def page_rank(graph):
    dict = nx.pagerank(graph, max_iter=50, alpha=0.9)
    return dict

def find_top_n_candidates(dict,n):
    sorted_dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_dict[0:n]

def part_of_speech_tag(candidates):
    tagged_candidates = nltk.pos_tag(candidates)
    new_candidates = []
    for candidate in tagged_candidates:
        if (candidate[1].startswith("NN")):
            new_candidates.append(candidate[0])
    return new_candidates

def ngramify(sentences, n_range: tuple):
    ngramified_doc = []
    for i in range(*n_range):
        for sentence in sentences:
            ngramified_doc += list(ngrams(sentence, i))
    return ngramified_doc


def get_candidates(document,ngram_range, candidateify = True):
    stop_words = stopwords.words('english')
    punctuation = string.punctuation

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
    # Lemmatize words
    wnl = nltk.WordNetLemmatizer()
    sentences = [[wnl.lemmatize(word) for word in sentence] for sentence in sentences]
    if candidateify:
        # Select parts of speech
        sentences = [part_of_speech_tag(sentence) for sentence in sentences]
        # Gramify document
        sentences = ngramify(sentences, ngram_range)
    return sentences

#Finds every sentence number all the words are in, every word in the document and every word's position in the document.
def word_occurences(document,n_gram):
    sentences = nltk.sent_tokenize(document)
    ngram_vectorizer = CountVectorizer(ngram_range=(n_gram))
    occurrences = ngram_vectorizer.fit_transform(sentences)
    words = ngram_vectorizer.get_feature_names()
    word_positions = {}
    for word in words:
        if " " in word:
            ws = word.split(" ")
            for w in ws:
                word_positions[w] = []
                word_positions[w].append(document.find(w))
        else :
            word_positions[word] = []
            word_positions[word].append(document.find(word))
    return occurrences, words, word_positions

def find_candidate_word_occurences(document,n_gram):
    candidates = get_candidates(document,n_gram,True)
    doc = get_candidates(document,n_gram,False)
    doc1 = " ".join(str(c)+'.' for c in doc)
    word_occurence,words,word_pos = word_occurences(doc1,n_gram)
    dict = {}
    #rows = the sentence number object i occurs in, object i can be in several
    #cols = the term i and the corresponding sentence  = rows[col] = raden
    rows,cols = word_occurence.nonzero()
    for candidate in candidates:
        candidate_sentences = []
        if len(candidate) == 1 and len(candidate[0]) > 1: # UNIGRAM
            candidate_index = words.index(candidate[0])
            indices = [index for index, value in enumerate(cols) if value == candidate_index]
            for i in range(0, len(indices)):
                candidate_sentences.append(rows[indices[i]])
            dict[candidate[0]] = candidate_sentences
        else:
            res = " ".join(str(c) for c in candidate)
            try :
                candidate_index = words.index(res)
                indices = [index for index, value in enumerate(cols) if value == candidate_index]
                for i in range(0, len(indices)):
                    candidate_sentences.append(rows[indices[i]])
                dict[candidate] = candidate_sentences
            except:
                candidate_sentences = 'NaN'
                #remove this if we dont want the NaN candidates to be in the dict
                #dict[candidate] = candidate_sentences

    return dict,word_pos



def find_edges(sentence_dict,word_pos_dict):
    edges = []
    candidates = list(sentence_dict.keys())
    for i in range(0,len(candidates)-1):
        if isinstance(candidates[i],str):
            candidate_pos1 = word_pos_dict[candidates[i]]
        else :
            words_in_candidate = [candidate for candidate in candidates[i]]
            candidate_pos1 = []
            for word in words_in_candidate:
                candidate_pos1.append(word_pos_dict[word])
        for j in range(i+1,len(candidates)):
            values = sentence_dict[candidates[i]]
            if isinstance(candidates[j], str):
                candidate_pos2 = word_pos_dict[candidates[j]]
            else:
                words_in_candidate = [candidate for candidate in candidates[j]]
                candidate_pos2 = []
                for word in words_in_candidate:
                    candidate_pos2.append(word_pos_dict[word][0])
            for val in values:
                for pos1 in candidate_pos1:
                    if isinstance(pos1, list):
                        pos1 = pos1[0]
                    if val in sentence_dict[candidates[j]] and pos1 not in candidate_pos2:
                        edges.append((candidates[i],candidates[j]))
    return edges


def e1(document,number_of_candidates):
    candidate_dict,word_pos_dict = find_candidate_word_occurences(document,(1,4))
    candidates = candidate_dict.keys()
    edges = find_edges(candidate_dict,word_pos_dict)
    G = make_graph(candidates,edges)
    #visualize_graph(G)
    ranked_candidates_dict = page_rank(G)
    top_n_candidates = find_top_n_candidates(ranked_candidates_dict,number_of_candidates)
    return top_n_candidates



def main():
    document = "The case, which has been bitterly contested for decades by Hindus and Muslims, centres on the ownership of the land in Uttar Pradesh state. Muslims would get another plot of land to construct a mosque, the court said. Many Hindus believe the site is the birthplace of one of their most revered deities, Lord Ram. Muslims say they have worshipped there for generations."
    document1 = "This is the document document document. 123 the document ##. This is a strange document. All the words get the same value document. The article that follows is ment as a tutorial for new students. The theme is nothing. Honestly, i think this is working. But we have to have a longer document. Therefore, i write this. So, lets go!"
    #print(get_candidates("This is . 123 the document ## :D  The article that follows is ment as a tutorial for new students to learn how to adapt to new documents. The theme is well-being."))
    #print(find_candidate_word_occurences(document,(1,4)))
    dict,word_pos_dict = find_candidate_word_occurences(document,(1,3))

    candidates = dict.keys()
    links = find_edges(dict,word_pos_dict)
    print('links')
    print(links)
    G = make_graph(candidates,links)
    #visualize_graph(G)
    dict = page_rank(G)
    print(find_top_n_candidates(dict,10))
    #print(n_gram_locator(document,3))

#if __name__ == '__main__':
#    main()


if __name__ == '__main__':
    print(e1("The case, which has been bitterly contested for decades by Hindus and Muslims, centres on the ownership of the land in Uttar Pradesh state. Muslims would get another plot of land to construct a mosque, the court said. Many Hindus believe the site is the birthplace of one of their most revered deities, Lord Ram. Muslims say they have worshipped there for generations.",10))
