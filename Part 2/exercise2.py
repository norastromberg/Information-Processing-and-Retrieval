#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Information Processing and Retrieval @IST 2019/2020
Course Project Part2
Exercise 2 - Improving the graph-ranking method
Group 23


"""
import networkx as nx
from gensim.models import KeyedVectors
from proj.part2.exercise1 import find_edges, make_graph, page_rank, find_top_n_candidates
from proj.part2.methods_for_e2.doc_prep_and_keyword_fetch import extract_specific_keywordlist, \
    fetch_most_relevant_keyphrases
from proj.part2.methods_for_e2.map import compute_AP
from proj.part2.methods_for_e2.store_and_load import load_data_pickle, store_data_pickle, load_all
from gensim.models import KeyedVectors



# Computing prior probability for each candidate based on their word lengths
def prior_prob_based_on_length(candidates_dict):
    personalization_dict = {}
    candidates = candidates_dict.keys()
    for candidate in candidates:
        char_count = 0
        for cand in candidate:
            char_count = char_count + len(cand)
        weight = char_count / len(candidate)
        personalization_dict.update({candidate: weight})
    return personalization_dict


# Assigning tfidf scores to candidates and creating a prior probability dictionary to use as
# personalization parameter in the page rank method
def prior_prob_with_tf_idf(candidate_dict, feature_score_dict):
    prior_tfidf_prob = {}
    for cand in candidate_dict.keys():
        for feature in feature_score_dict.keys():
            if isinstance(cand, str):
                if cand == feature:
                    prior_tfidf_prob.update({cand: feature_score_dict.get(cand)})
                else:
                    prior_tfidf_prob.update({cand: 0})
            if isinstance(cand, tuple):
                sum_cand_score = 0
                for c in cand:
                    if c in feature_score_dict.keys():
                        sum_cand_score = sum_cand_score + feature_score_dict.get(c)
                avg_cand_score = sum_cand_score/len(cand)
                prior_tfidf_prob.update({cand: avg_cand_score})
    return prior_tfidf_prob


# computing redirecting to edge weight computation
def compute_edge_weights(candidates_dict, condition, word_pos_dict):
    edge_weights = {}
    edges = find_edges(candidates_dict, word_pos_dict=word_pos_dict)
    if condition == 'co-occurrences':
        edge_weights = edge_weights_co_occurrences(edges)
    elif condition == 'word_similarity':
        edge_weights = edge_weights_word_similarity(edges)
    return edge_weights


# Computing edge weights based on word similarity using pre-trained word embeddings
def edge_weights_word_similarity(edges):
    edge_weights = {}
    word_vectors = load_data_pickle('word_vector_model.data')
    count_words_not_in_vocab = 0
    count_unique_edges = 0
    unique_edges = []

    for edge in edges:
        if edge not in unique_edges:
            count_unique_edges = count_unique_edges + 1
            if isinstance(edge[0], str) and isinstance(edge[1], str):
                if edge[0] in word_vectors and edge[1] in word_vectors:
                    similarity = word_vectors.similarity(edge[0], edge[1])
                else:
                    similarity = 0
                    count_words_not_in_vocab = count_words_not_in_vocab + 1
                edge_weights.update({edge: similarity})
            elif isinstance(edge[0], str) and isinstance(edge[1], tuple):
                sim_sum = 0
                for e in edge[1]:
                    if edge[0] in word_vectors and e in word_vectors:
                        sim_sum = sim_sum + word_vectors.n_similarity(edge[0], e)
                    similarity = sim_sum/len(edge[1])
            else:
                edge0 = []
                for e in edge[0]:
                    if e in word_vectors:
                        edge0.append(e)
                edge1 = []
                for e in edge[1]:
                    if e in word_vectors:
                        edge1.append(e)
                if edge1 == [] or edge0 == []:
                    similarity = 0
                else:
                    similarity = word_vectors.n_similarity(edge0, edge1)
            edge_weights.update({edge: similarity})
            unique_edges.append(edge)
    return edge_weights


# Computing edgeweights for edges based on how many times a candidate pair appears in the same sentence
def edge_weights_co_occurrences(edges):
    edge_weights = {}
    for edge in edges:
        if edge in edge_weights.keys():
            nr_of_co_occ = edge_weights.get(edge)
            edge_weights.update({edge: nr_of_co_occ + 1})
        else:
            edge_weights.update({edge: 1})
    return edge_weights


# Creates a graph of the candidates and executes regular page rank from exercise 1
def regular_page_rank(candidates_dict, word_pos_dict):
    links = find_edges(candidates_dict, word_pos_dict)
    G = make_graph(candidates_dict, links)
    ranked = page_rank(G)
    return ranked


# Computing page rank with a prior rpobability for all candidates as personalization input to the nodes in the graph
def page_rank_with_personalization(candidates_dict, personalization, word_pos_dict):
    links = find_edges(candidates_dict, word_pos_dict)
    candidates = candidates_dict.keys()
    G = make_graph(candidates, links)
    rank_with_personalization = nx.pagerank(G, max_iter=50, alpha=0.9, personalization=personalization)
    return rank_with_personalization


# Creating a graph with candidates as nodes, co-occurrences in sentences as edges.
# Input: Candidates_dictionary and Edge_weights dictionary. Doing pagerank on this created graph and returns it
def page_rank_with_edges(candidates_dict, edge_weights):
    G = nx.Graph()
    candidates = list(candidates_dict.keys())
    for i in range (0, len(candidates)-1):
        for j in range(i + 1, len(candidates)):
            if (candidates[i], candidates[j]) in edge_weights.keys():
                w = edge_weights.get((candidates[i], candidates[j]))
                G.add_edge(candidates[i], candidates[j], weight=w)
    p_rank = nx.pagerank(G)
    return p_rank


# Finding top candidates from a specific document based on a userspecified criteria. Criteria can be 'word_similarity',
# 'co-occurrences',  'tfidf', 'n_gram_length' or 'regular'. Returns a sorted dictionary with the top n best ranked candidates
# based on given criteria
def get_top_candidates(criteria, n_top, candidates_dict, features_score_dict, word_pos_dict):
    if criteria == 'n_gram_length':
        personalization_dict = prior_prob_based_on_length(candidates_dict)
        ranked = page_rank_with_personalization(candidates_dict, personalization_dict, word_pos_dict)
    elif criteria == 'tfidf':
        personalization_dict = prior_prob_with_tf_idf(candidates_dict, features_score_dict)
        ranked = page_rank_with_personalization(candidates_dict, personalization_dict, word_pos_dict)
    elif criteria == 'co-occurrences':
        edge_weights = compute_edge_weights(candidates_dict, criteria, word_pos_dict)
        ranked = page_rank_with_edges(candidates_dict, edge_weights)
    elif criteria == 'word_similarity':
        edge_weights = compute_edge_weights(candidates_dict, criteria, word_pos_dict)
        ranked = page_rank_with_edges(candidates_dict, edge_weights)
    elif criteria == 'regular':
        ranked = regular_page_rank(candidates_dict, word_pos_dict)
    else:
        ranked = None
    #Sorts and select the n_top of the ranked dictionary
    top_results = find_top_n_candidates(ranked, n_top)
    return top_results


# Computing MAP for all criterions with different values for top n candidates. This is to use the
# lists that it returns in the plotting in the next method
def MAP_lists(doc_id_dict, keyword_dict, features_score_dict, all_cands_dict, all_word_pos_dict):
    reg_list = []
    len_list = []
    tf_list = []
    co_list = []
    wsim_list = []
    n_list = []
    for n in range(5, 30, 5):
        print(n)
        print('REG')
        reg = compute_MAP(doc_id_dict, keyword_dict,'regular', features_score_dict, all_cands_dict, all_word_pos_dict, n)
        reg_list.append(reg)
        print('LEN')
        len = compute_MAP(doc_id_dict, keyword_dict,'n_gram_length', features_score_dict, all_cands_dict, all_word_pos_dict, n)
        len_list.append(len)
        print('TFIDF')
        tfidf = compute_MAP(doc_id_dict, keyword_dict, 'tfidf', features_score_dict, all_cands_dict,all_word_pos_dict, n)
        tf_list.append(tfidf)
        print('CO-OCC')
        co = compute_MAP(doc_id_dict, keyword_dict, 'co-occurrences', features_score_dict, all_cands_dict, all_word_pos_dict, n)
        co_list.append(co)
        print('W-SIM')
        wsim = compute_MAP(doc_id_dict, keyword_dict, 'word_similarity', features_score_dict, all_cands_dict, all_word_pos_dict, n)
        wsim_list.append(wsim)
        n_list.append(n)

    return reg_list, len_list, tf_list, co_list, wsim_list, n_list


# Computing MAP for all documents with a given criteria. Criteria can be 'word_similarity', 'co-occurrences',  'tfidf'
# 'n_gram_length' or 'regular'
def compute_MAP(doc_id_dict, keyword_dict, criteria, tfidf_score_dict, all_cands_dict, all_word_pos_dict, top_n):
    sum_AP = 0
    count = 0
    outliers = 0
    for doc_name in doc_id_dict:
        cand_dict = all_cands_dict.get(doc_name)
        word_pos_dict = all_word_pos_dict.get(doc_name)
        top_cands = get_top_candidates(criteria, top_n , cand_dict, tfidf_score_dict, word_pos_dict)
        solution_keywords = extract_specific_keywordlist(keyword_dict, doc_name)
        top_cand_string_list = from_tuple_to_string(top_cands)
        AP = compute_AP(top_cand_string_list, solution_keywords)
        sum_AP = sum_AP + AP
        count = count + 1
        if AP == 0:
            outliers = outliers + 1
    MAP = sum_AP/count
    print('MAP for ', criteria, ' = ', MAP)
    return MAP


# Plotting the MAP for each approach against different numbers of top_n to see which n that gives the best MAP overall
def plot_MAPs(top_ns, reg_maps, length_maps, tfidf_maps, co_occ_maps, word_sim_maps):
    from matplotlib.pylab import plt
    plt.plot(top_ns, reg_maps, label="Regular Page Rank")
    plt.plot(top_ns, length_maps, label="Prior prob - wordlength ")
    plt.plot(top_ns, tfidf_maps, label="Prior prob - TF-IDF score")
    plt.plot(top_ns, co_occ_maps, label="Edge weights - #co-occurrences")
    plt.plot(top_ns, word_sim_maps, label="Edge weights - word similarity")
    plt.legend()
    plt.title("MAP, different PR approaches with different number of top_n", fontsize=14, fontweight='bold')
    plt.xlabel("Top_n candidates")
    plt.ylabel("MAP")
    plt.show()


# Storing keyed vectors from the pre-trained word embeddings. This function is necessary to execute once
# before running the Page Rank approach with edge weights based on word similarity. And to run this it is
# necessary to download the wiki-news-300d-1M.vec file and use the filepath as input to this function
def store_keyed_vectors_model(filepath):
    model = KeyedVectors.load_word2vec_format(filepath, binary=False)
    store_data_pickle(model, 'word_vector_model.data')
    print('Stored model in file:  word_vector_model.data')
    return model


# Creating strings of tuples
def from_tuple_to_string(top_cand_dict):
    cand_list = []
    for candidate in top_cand_dict:
        cand = candidate[0]
        if isinstance(cand, str):
            cand_list.append(cand)
        else:
            cand_str = ''
            for i in range(len(cand)):
                cand_str += cand[i]
                if i < len(cand)-1:
                    cand_str += ' '
            cand_str.strip()
            cand_list.append(cand_str)
    return cand_list



# Exercise 2
def e2():

    # With this call the pre-trained word embeddings file will be created as a model and stored
    # store_keyed_vectors_model('wiki-news-300d-1M.vec')

    corpus, corpus_idx_dict, xml_files, target_idx_dict, all_cands_dicts, features_score_dict, all_word_pos_dict = load_all()
    keyword_dict = fetch_most_relevant_keyphrases('proj/part2/data/train_combined.json')

    # Criteria can be: regular, n_gram_length, tfidf, co-occurrences or word_similarity
    print('Computing MAP for Page Rank with different criterias...')
    compute_MAP(corpus_idx_dict, keyword_dict,'regular', features_score_dict, all_cands_dicts, all_word_pos_dict, 15)
    compute_MAP(corpus_idx_dict, keyword_dict,'n_gram_length', features_score_dict, all_cands_dicts, all_word_pos_dict, 15)
    compute_MAP(corpus_idx_dict, keyword_dict,'tfidf', features_score_dict, all_cands_dicts, all_word_pos_dict, 15)
    compute_MAP(corpus_idx_dict, keyword_dict,'co-occurrences', features_score_dict, all_cands_dicts, all_word_pos_dict, 15)
    compute_MAP(corpus_idx_dict, keyword_dict,'word_similarity', features_score_dict, all_cands_dicts, all_word_pos_dict, 15)

    # Plotting:
    #reg_list, len_list, tf_list, co_list, wsim_list, n_list = MAP_lists(corpus_idx_dict, keyword_dict, features_score_dict, all_cands_dicts, all_word_pos_dict)
    #plot_MAPs(n_list, reg_list, len_list, tf_list, co_list, wsim_list)


if __name__ == '__main__':
    e2()
