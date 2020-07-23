# Copyright 2018 CVTE . All Rights Reserved.
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import nltk
import pickle
import math
import time


class Text_judge:
    def __init__(self, path, n_gram):
        self.dict_ngram, self.vocab_size, self.lexcion_size = pickle.load(open(path, 'rb'))
        self.n_gram = n_gram

    def getprobs(self, words):
        if len(words) == 1:
            k = float(self.dict_ngram[tuple(words)] + 1) / float(self.vocab_size + self.lexcion_size)
            return k
        elif self.dict_ngram[tuple(words)] != 0:
            probs = (self.dict_ngram[tuple(words)] * 1.0) / self.dict_ngram[tuple(words[:-1])]
        else:
            # if len(words) > 1:
            backprob = self.getprobs(words[1:])
            probs = 0.4 * backprob
            # else:
            #     backprob = getprobs(words, dict_ngram, vocab_size, lexcion_size)
            #     probs = 0.4 * backprob
        return probs

    def get_log_prob(self, words):
        probs = self.getprobs(words)
        return math.log(probs)

    def sentence_log_prob(self, words):
        # 获取一个句子中，所有的ngram概率返回一个list
        logprobs = []
        if len(words) <= self.n_gram:
            # 开始的几个单词，不够ngram个词，特殊处理
            logprobs.append(self.get_log_prob(words))
        else:
            wordngrams = nltk.ngrams(words, self.n_gram)
            for wordTrigram in wordngrams:
                logprobs.append(self.get_log_prob(wordTrigram))
        return logprobs


if __name__ == "__main__":


    judge_object = Text_judge("lm_dict.bin",5)

    all_text = ["wqewq wqe wq ewq ewq e",
        "What time it is now?",
        "Some people make your laugh a little louder, your smile a little brighter and your life a little better.",
        "dsjkfignvfidvnv jfdfnoijjkmvfk dsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfkdsjkfignvfidvnv jfdfnoijjkmvfk",
        "dsffsdfsdfs",
        "dsffsdfsdfs",
        "DFdcvfdgfvfbvbdgfbfg",
        "how lonng have beenyoubeenyou herle?",
        "how lonngvjykghkwag have beenyou here?",
        "dsffasl like to ttt learn",
         "I like to to learn English",
         "I like to learn English",
                "ihtn vjkjvn njknfoew njjsojjehffffffffff",
                "weq",
                "wwwwwwwwwwwwwwwww",
                "tttttttttttttttt"]

    t = time.time()
    for i in range(10000):
        for text in all_text:
            prob = judge_object.sentence_log_prob(list(text))
            print(sum(prob) / len(text))
    print("time:", time.time() - t)

