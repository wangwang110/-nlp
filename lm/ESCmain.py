# Copyright 2018 CVTE . All Rights Reserved.
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from satistics_lm import LangModel
import nltk
import pickle



class Text_judge:
    def __init__(self, path, n_gram):
        self.lm = pickle.load(open(path, 'rb'))
        self.n_gram = n_gram

    def get_log_prob(self, words):
        logprob = self.lm.logprob(tuple(words))
        ## log1==0
        ## 值越大，越不是合理的词
        return logprob

    def get_raw_prob(self, words):
        prob = self.lm.prob(tuple(words))
        return prob

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
    judge_object = Text_judge("./lm5.bin",5)

    all_text = ["wqewq wqe wq ewq ewq e",
        "What time it is now?",
        "Some people make your laugh a little louder, your smile a little brighter and your life a little better.",

        "dsjkfignvfidvnv jfdfnoijjkmvfk",
        "dsffsdfsdfs",
        "dsffsdfsdfs",
        "DFdcvfdgfvfbvbdgfbfg",
        "how lonng have beenyoubeenyou herle?",
        "how lonngvjykghkwag have beenyou here?",
        "dsffasl like to ttt learn",
         "I like to to learn English",
         "I like to learn English",
                "ihtn vjkjvn njknfoew njjsojjehffffffffff",
                "weqqqqqtqtqq",
                "wwwwwwwwwwwwwwwww",
                "tttttttttttttttt"]

    for text in all_text:
        prob = judge_object.sentence_log_prob(list(text))
        print(sum(prob)/len(text))

