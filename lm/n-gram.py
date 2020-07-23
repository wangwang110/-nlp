# Copyright 2018 CVTE . All Rights Reserved.
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import nltk
import pickle
import math


def get_ngram_dict(datafile, ngram=5):
    n = 0
    lexicon = set()
    lines = open(datafile, 'r').readlines()
    dict_ngram = nltk.FreqDist()
    for line in lines:
        words = line.strip().lower().split(" ")
        for i in range(1, ngram + 1):
            wordNGrams = nltk.ngrams(words, i)
            for wordNGram in wordNGrams:
                dict_ngram[wordNGram] += 1
                if i == 1 and wordNGram[0] != "<s>" and wordNGram[0] != "</s>":
                    lexicon.add(wordNGram)
                    n += 1
    lexcion_size = len(lexicon)
    return dict_ngram, n, lexcion_size


def getprobs(words, dict_list, vocab_size, lexcion_size):
    ngram_len = len(words)
    dict_ngram = dict_list[ngram_len - 1]
    key_word = str(tuple(words))

    if (ngram_len == 1):
        if key_word in dict_ngram.keys():
            k = float(dict_ngram[key_word] + 1) / float(vocab_size + lexcion_size)
        else:
            k = float(1) / float(vocab_size + lexcion_size)
        return k

    elif key_word in dict_ngram.keys():
        back = dict_list[ngram_len - 2]
        freq = dict_ngram[key_word]
        backfreq = back[str(tuple(words[:-1]))]
        probs = (freq * 1.0) / backfreq
    else:
        backprob = getprobs(words[1:], dict_list, vocab_size, lexcion_size)
        probs = 0.4 * backprob
    return probs


def getlogprob(words, dict_list, vocab_size, lexcion_size):
    probs = getprobs(words, dict_list, vocab_size, lexcion_size)
    return math.log(probs)


def sentence_log_prob(words, dict_list, vocab_size, lexcion_size, n_gram=3):
    ##获取一个句子中，所有的ngram概率返回一个list
    logprobs = []
    if (len(words) <= n_gram):
        logprobs.append(getlogprob(words, dict_list, vocab_size, lexcion_size))
    else:
        wordngrams = nltk.ngrams(words, n_gram)
        for wordTrigram in wordngrams:
            # print(wordTrigram)
            logprobs.append(getlogprob(wordTrigram, dict_list, vocab_size, lexcion_size))
    return logprobs


def loadfile(ngram):
    re_list = []
    for i in range(1, ngram + 1):
        ngram_name = nltk.FeatDict()
        f1 = open("5ngram_" + str(i) + ".txt", "r")
        words = f1.readlines()
        f2 = open("5frency_" + str(i) + ".txt", "r")
        fc = f2.readlines()
        for k, v in zip(words, fc):
            ngram_name[k.strip()] = int(v.strip())
        f1.close()
        f2.close()
        re_list.append(ngram_name)
    return re_list


if __name__ == "__main__":
    datafile = "./processdata.txt"
    ngram = 5
    # dict_ngram, vocab_size, lexcion_size = get_ngram_dict(datafile, ngram)
    #
    # for i in range(1, ngram + 1):
    #     f1 = open("5ngram_" + str(i) + ".txt", "w")
    #     f2 = open("5frency_" + str(i) + ".txt", "w")
    #     for k, v in dict_ngram.items():
    #         if len(list(k)) == i:
    #             f1.write(str(k) + '\n')
    #             f2.write(str(v) + "\n")
    #     f1.close()
    #     f2.close()
    #
    # f = open("size.txt", "w")
    # f.write(str(vocab_size) + ' ' + str(lexcion_size))
    # f.close()

    dict_list = loadfile(ngram)
    vocab_size, lexcion_size = open("size.txt", "r").read().strip().split(" ")

    words = "Keeping the Secret of Genetic the ".lower().split()
    print(sentence_log_prob(words, dict_list, vocab_size, lexcion_size, 5))
