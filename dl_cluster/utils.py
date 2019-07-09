# -*- coding: utf-8 -*-
import os
import re
import sys
import settings as config
import numpy as np
import codecs
import pandas as pd
from naming_lib import colname as colname


# Generate word vacabulary
def create_vocabulary(data):
    vocab = {}
    count = 0
    for line in data:
        count +=1
        if count % 10000 == 0:
            print("Processed data：%d" % count)
        tokens = line.split(" ")
        for word in tokens:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
    vocab = {key: value for key, value in vocab.items() if value > config.filter_words_frequent}

    with open(config.vocabulary_freq, 'w',encoding='utf-8') as f:
        f.write('{0},{1},{2},{3},{4}\n'.format(colname.col_out1, colname.col_out2, colname.col_out3, colname.col_out4, colname.col_out5))
        for word in sorted(vocab, key=vocab.get, reverse=True):
            relative_sentence,near_words,key_words,number_bugs = get_words_relative_sentence_and_words(word, data)
            f.write('{0},{1},{2},{3},{4}\n'.format(word, vocab[word], ";".join(near_words), ";".join(key_words), number_bugs))


def wordcount(total_list):
    # string preprocessing
    count_dict = {}
    for token_ist in total_list:
        for token in token_ist:
            if token in count_dict.keys():
                count_dict[token] = count_dict[token] + 1
            else:
                count_dict[token] = 1
    # Order word frequency descendingly
    count_list = sorted(count_dict, key=count_dict.get, reverse=True)
    return count_list


# Generate keyword analysis report
def get_words_relative_sentence_and_words(key_word,total_data):
    '''
    Compute keyword related sentences
    :param key_word:
    :param total_data:
    :return:
    '''
    relative_sentence = []

    # find out all sentences including keyword by interating all data
    for sentence in total_data:
        split_sentence = sentence.split(" ")
        if key_word in split_sentence:
            relative_sentence.append(split_sentence)

    # Calculate word count
    count_list_top_5 = wordcount(relative_sentence)[:5]

    # Compute number of sentences with top k keyword, select the representative sentences as the ones with most keywords
    relative_sentence_num_of_top_words = []
    for sentence in relative_sentence:
        include_words = [word for word in list(set(sentence)) if word in count_list_top_5]
        relative_sentence_num_of_top_words.append(len(include_words))

    # Get index of representative sentence
    maxindex = np.argmax(np.array(relative_sentence_num_of_top_words))
    output_sentence = relative_sentence[maxindex]

    # get 5 words around keyword
    top_3_phrase = []
    top_5_relative_sentence_index = (np.argsort(np.array(relative_sentence_num_of_top_words)).tolist()[::-1])[:5]
    for index in top_5_relative_sentence_index:
        key_word_index = relative_sentence[index].index(key_word)
        position_start = max(0,key_word_index - 2)
        position_end = min(len(output_sentence),key_word_index + 3)

        words_between_keyword =" ".join(relative_sentence[index][position_start:position_end])
        top_3_phrase.append(words_between_keyword)

    #out_relative_words = [word for word in output_sentence if word in count_list_top_5]
    return  output_sentence,top_3_phrase, set(count_list_top_5), len(relative_sentence)
