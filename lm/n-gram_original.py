# Copyright 2018 CVTE . All Rights Reserved.
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import nltk
import pickle
import math
from collections import defaultdict

def get_ngram_dict(sentences, ngram):
    """
    字符级别的统计语言模型
    :param sentences:
    :param ngram:
    :return:
    """
    vocabulary = []
    dict_ngram = nltk.FreqDist()
    final_dict_ngram = {}
    for sent in sentences:
        words = list(sent)
        vocabulary.extend(words)
        for i in range(1, ngram + 1):
            wordNGrams = nltk.ngrams(words, i)
            for wordNGram in wordNGrams:
                dict_ngram[wordNGram] += 1

    for item_key in dict_ngram.keys():
        num = len(item_key)
        if num not in final_dict_ngram:
            final_dict_ngram[num] = nltk.FreqDist()
            final_dict_ngram[num][item_key] = dict_ngram[item_key]
        else:
            final_dict_ngram[num][item_key] = dict_ngram[item_key]

    vocab_size = len(vocabulary)
    lexcion_size = len(set(vocabulary))
    final_dict_ngram["vocab_size"] = vocab_size
    final_dict_ngram["lexcion_size"] = lexcion_size

    return final_dict_ngram

def getprobs(words, dict_ngram, vocab_size, lexcion_size):
    if len(words) == 1:
        k = float(dict_ngram[tuple(words)] + 1) / float(vocab_size + lexcion_size)
        return k
    elif dict_ngram[tuple(words)] != 0:
        probs = (dict_ngram[tuple(words)] * 1.0) / dict_ngram[tuple(words[:-1])]
    else:
        # if len(words) > 1:
        backprob = getprobs(words[1:], dict_ngram, vocab_size, lexcion_size)
        probs = 0.4 * backprob
        # else:
        #     backprob = getprobs(words, dict_ngram, vocab_size, lexcion_size)
        #     probs = 0.4 * backprob
    return probs


def getlogprob(words, dict_ngram, vocab_size, lexcion_size):
    probs = getprobs(words, dict_ngram, vocab_size, lexcion_size)
    return math.log(probs)


if __name__ == "__main__":
    path = "./train.all.trg"
    n_gram = 5
    sentences = []
    import codecs
    lines = codecs.open(path, 'r', "utf-8").readlines()
    for line in lines:
        sentences.append(line.strip())

    final_dict_ngram = get_ngram_dict(sentences, n_gram)

    pickle.dump(final_dict_ngram, open("lm_dict_tmp.bin", "wb"))
    # words = "Keeping the Secret of Genetic Testing".lower().split()
    # print(getlogprob(words, dict_ngram, vocab_size, lexcion_size))
