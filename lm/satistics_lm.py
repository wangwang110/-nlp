# Copyright 2018 CVTE . All Rights Reserved.
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import nltk
import math
import pickle
import sys

class LangModel:
    def __init__(self, order, alpha, sentences):
        self.index = 0
        self.order = order
        self.alpha = alpha
        if order > 1:
            self.backoff = LangModel(order - 1, alpha, sentences)
            self.lexicon = None
        else:
            self.backoff = None
            self.n = 0

        self.ngramFD = nltk.FreqDist()
        # 元组列表，（单词，出现的频次）
        lexicon = set()
        for sentence in sentences:
            self.index += 1
            words = list(sentence)
            wordNGrams = nltk.ngrams(words, order)
            # 获得所有n元组合
            for wordNGram in wordNGrams:
                self.ngramFD[wordNGram] += 1
                if order == 1:
                    lexicon.add(wordNGram)
                    # 单词表,不重复
                    self.n += 1
                    # 单词个数，可重复
        self.v = len(lexicon)
        # 单词表

    def logprob(self, ngram):
        t = self.prob(ngram)
        return math.log(t)

    def prob(self, ngram):
        # 5-gram语言模型
        current = self
        while current.order > len(ngram):
            current = current.backoff
        if current.backoff != None:
            freq = current.ngramFD[ngram]
            backoffFreq = current.backoff.ngramFD[ngram[:-1]]
            if freq == 0:
                ## 五元组为0
                if len(ngram) > 1:
                    # 如果五元组不存在，从开头减少一个
                    backprob = current.backoff.prob(ngram[1:])
                    return current.alpha * backprob
                else:
                    # 不能再减少下去了
                    backprob = current.backoff.prob(ngram)
                    return current.alpha * backprob
            else:
                ##五元组不为0，四元组肯定大于0
                return freq / backoffFreq
        else:
            ## current.backoff == None 此时是unigram模型
            # laplace smoothing to handle unknown unigrams
            ## 以上两个值不一样嘛？用实际训练数据进行测试下
            k = float(float(current.ngramFD[ngram] + 1) / float(current.n + current.v))
            return k


def train(path):
    # if os.path.isfile("lm3.bin"):
    #     return
    n_gram = 5
    sentences = []
    lines = open(path, 'r').readlines()
    for line in lines:
        sentences.append(line.strip())
    lm = LangModel(n_gram, 0.4, sentences)
    # 0.4 no use
    print("has trained!!")
    pickle.dump(lm, open("lm5.bin", "wb"))


if __name__ == "__main__":
    path = sys.argv[1]
    train(path)

