# encoding=utf-8
from __future__ import division
import math
import os.path
import nltk
import string
import numpy as np
import distance
import lcs
import cPickle
import pickle
import time

import copy

import time
import numpy as np
import codecs
import argparse
import math
import sys


# For KenLM features
sys.path.insert(0, 'lib/kenlm_python/')
import kenlm



start_error="<span class=\"jiaodui-spell-error-3\">"
end_error="</span>"

start_correct="<span class=\"jiaodui-spell-error-1\">"
end_correct="</span>"
replace_threshold = 0.3
n_gram = 5
dir_path=os.path.dirname(os.path.realpath(__file__))



class LM:
    def __init__(self, name, path, normalize=False, debpe=False):
        self.path = path
        c = kenlm.Config()
        c.load_method = kenlm.LoadMethod.LAZY
        self.model = kenlm.Model(path, c)
        self.name = name
        self.normalize = normalize
        self.debpe = debpe
        logger.info('Intialized ' + str(self.model.order) + "-gram language model: " + path)

    def get_name(self):
        return self.name

    def get_score(self, candidate):
        lm_score = self.model.score(candidate)
        return lm_score


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
        lexicon = set()
        # set NUll
        for sentence in sentences:
            self.index += 1
            words = sentence.split(" ")
            wordNGrams = nltk.ngrams(words, order)
            for wordNGram in wordNGrams:
                self.ngramFD[wordNGram] += 1
                if order == 1 and wordNGram[0] != "<s>" and wordNGram[0] != "</s>":
                    lexicon.add(wordNGram)
                    self.n += 1
        self.v = len(lexicon)

    def logprob(self, ngram):
        t = self.prob(ngram)
        return math.log(t)

    def prob(self, ngram):
        current = self
        while current.order > len(ngram):
            current = current.backoff

        # current.order==len(ngram)

        if current.backoff != None:

            freq = current.ngramFD[ngram]
            backoffFreq = current.backoff.ngramFD[ngram[:-1]]
            if freq == 0:
                if len(ngram) > 1:
                    backprob = current.backoff.prob(ngram[1:])
                    return current.alpha * backprob
                else:
                    backprob = current.backoff.prob(ngram)
                    return current.alpha * backprob

            else:
                if backoffFreq > 0:
                    return freq / backoffFreq
                else:
                    return freq / current.n

        else:
            # laplace smoothing to handle unknown unigrams
            # print current.n, "-=-=-=-=-"
            k = float(float(current.ngramFD[ngram] + 1) / float(current.n + current.v))
            return k



def load_model():
    lm1_path = os.path.join(dir_path, "concat-train.trie")
    return open(lm1_path, 'r')



def load_files():
    print "loading"
    dictionary_path = os.path.join(dir_path, "dictionary_youdao_1.txt")
    dictionary_object = open(dictionary_path)
    dictionary = []
    for line in dictionary_object:
        line = line.strip().replace("\n", "").replace("\r", "")
        dictionary.append(line)
    lines = os.path.join(dir_path, "candidates.pkl")
    s = open(lines, 'r')
    candidates = pickle.load(s)
    f2true_path = os.path.join(dir_path, "f2true.pkl")
    temp = open(f2true_path, 'rb')
    f2true = pickle.load(temp)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    lm1=load_model()
    return dictionary,candidates,f2true,tokenizer,lm1

start=time.time()
dictionary,candidates,f2true,tokenizer,lm1=load_files()
print("load time",time.time()-start)

def get_log_prob(words):
    #
    # print (tuple(words))
    logprob = lm1.get_score(tuple(words)) * -1
    ## log1==0
    ## 值越大，越不是合理的词
    return logprob


def get_raw_prob(words):
    prob = lm1.prob(tuple(words))
    return prob


def sentence_log_prob(words):
    ##获取一个句子中，所有的ngram概率返回一个list
    logprobs = []
    if (len(words) <= n_gram):
        ##开始的几个单词，不够ngram个词，特殊处理
        logprobs.append(get_log_prob(words))
    else:
        wordngrams = nltk.ngrams(words, n_gram)
        for wordTrigram in wordngrams:
            logprobs.append(get_log_prob(wordTrigram))
    return logprobs




def splitSentence(paragraph):
    ## nltk自带的分句文件，以句号分

    sentences = tokenizer.tokenize(paragraph)
    return sentences



def get_candidates(word):
    ## 获取一个词的候选集列表，根据分值

    candidates_words = []

    if word not in candidates.keys():
        ## 词不在离线计算的文件中，重新计算加入

        edits1 = distance.edits1(word)
        edits2 = distance.edits2(word)


        if word in f2true:
            edits1 = edits1 | set(f2true[word])

        dist1, dist2 = distance.known(edits1), distance.known(edits2)
        dist2 = dist2 - dist1 & dist2

        dist1 = list(dist1)
        dist2 = list(dist2)

        for w in dist1:
            candidates[word].append(
                (w, 1, lcs.find_lcsubstr(w, word) / len(word), lcs.find_lcseque(w, word) / len(word)))
        for w in dist2:
            candidates[word].append(
                (w, 0.5, lcs.find_lcsubstr(w, word) / len(word), lcs.find_lcseque(w, word) / len(word)))

    for temp in candidates[word]:
        if len(word) <= 1:
            continue
        elif len(word) == 2 and temp[1] == 0.5:
            continue
            # 编辑距离大于word长度，不作为候选词
        score = temp[1] * 2 + 0.7 * temp[2] + 0.5 * temp[3]

        if (temp[0][0] != word[0]):
            ##首字母相同
            score = 0.5 * score

        if (score > 1.5):
            candidates_words.append((temp[0], score))

    return candidates_words


def replace(words, index, isTrue=False):
    original_word = words[index]
    #original_word=original_word.lower()
    ## replace(sentence,words[i],i)
    candidates_words = get_candidates(words[index].lower())

    if (len(candidates_words) == 0):
        ##没有候选单词返回原有的单词，不改
        return original_word

    candidate_probs = {}
    start = max(0, index - n_gram + 1)
    end = -1 if len(words) - 1<index + n_gram-1 else index + n_gram

    ##避免越界


    for word in candidates_words:
        words[index] = word[0]
        ##替换原词为候选词，计算概率
        if end==-1:
            temp = words[start:]
        else:
            temp = words[start:end]
        # print "&&&"
        # print temp
        # print words
        # print start
        # print end
        # print index
        logprobs = sentence_log_prob(temp)
        ###logprobs=1
        candidate_probs[word[0]] = np.mean(logprobs)-word[1]
        # print (np.mean(logprobs)+1)*word[1]
        ###求和也可以

    result = min(candidate_probs.items(), key=lambda s: s[1])
    # (key,value)
    if original_word.islower():
        return result[0]
    elif original_word.isupper():
        return result[0].upper()
    else:
        return result[0].capitalize()




def process_txt(paragraph):

    N=[14,5,5,5,5]

    all_the_text = paragraph

    all_the_text=all_the_text.replace("-", " ")
    #页面输出的错误字符串
    display_text_error=all_the_text
    #页面输出的正确字符串
    display_text_correct = all_the_text

    len_error=len(start_error+end_error)
    len_correct=len(start_correct+end_correct)

    list_sentence=splitSentence(all_the_text)
    all_position=[]
    sentence_index=0

    for sentence in list_sentence:
        temp_sentence_index = all_the_text.find(sentence, sentence_index)
        list_sentence_word = nltk.word_tokenize(sentence.replace("can't","cann't").replace("won't","will not").replace("shan't","shall not"))
        word_index = 0
        temp_position = []
        for word in list_sentence_word:
            #print word
            temp_index=sentence.find(word,word_index)
            temp_position.append(temp_index+temp_sentence_index)
            if temp_index==-1:
               continue
            else:
                word_index=temp_index+len(word)
        all_position.append(temp_position)
        if temp_sentence_index == -1:
            continue
        else:
            sentence_index = temp_sentence_index + len(sentence)
    error_list=[]
    error_count=0
    bias=0
    for sentence_position_list,sentence in zip(all_position,list_sentence):
        list_sentence_word = nltk.word_tokenize(sentence.replace("can't","cann't").replace("won't","will not").replace("shan't","shall not"))
        i=-1
        for temp_word, temp_position in zip(list_sentence_word,sentence_position_list):
            i = i + 1
            if temp_word.isupper():
                #print"******"
                #print temp_word
                continue

            if i!=0 and not temp_word.decode("utf8")[0].islower():
                #print"******"
                #print temp_word
                continue


            if temp_word.lower() in dictionary:
                #print"******"
                #print temp_word
                continue
            elif temp_word not in string.punctuation and temp_word !="``" and temp_word !="''":
                correct_word = replace(copy.deepcopy(list_sentence_word), i, False)
                if(correct_word!=temp_word):
                     temp_false={}
                     temp_false["misspell"]=temp_word
                     temp_false["correct"]=correct_word
                     temp_false["position_start"]=temp_position
                     temp_false["position_end"] = temp_position+len(temp_word)
                     error_list.append(temp_false)

                     display_text_error=display_text_error[0:error_count*len_error+temp_position]+start_error+temp_word+end_error+display_text_error[error_count*len_error+temp_position+len(temp_word):]
                     display_text_correct = display_text_correct[0:error_count * len_correct + temp_position+bias] + start_correct + correct_word + end_correct + display_text_correct[error_count * len_correct+ temp_position+bias + len(temp_word):]
                     bias = bias + len(correct_word) - len(temp_word)
                     error_count=error_count+1
    error_dictionary={}
    error_dictionary["list"]=error_list
    return  display_text_error,display_text_correct,error_dictionary
if __name__ == '__main__':
    #Convert_txt_to_pkl("dictionary/vocab.txt")
    #nltk.download()
    # lm1=load_model()
    # process_txt_path = os.path.join(dir_path, "wiki_1")
    # print process_txt(process_txt_path,lm1)[0]
    # print "\n"
    # print process_txt(process_txt_path,lm1)[1]

    time1=time.time()
    print process_txt('''Love is everywhere. We have love from our parents and friends, which makes us become stronger. Without love, we can't survive long, or we will just like the walking dead. So many great people have owed their success to the love from families, but the love for nature always forgotten by the public.
Love from Beijin and frieds are praised in so many works. Like the great poets, they wrote so many famous works to give applause to the people who suport them all the time. In the movies and TV shows, family love and friendship occupy most themes. They shows the great power of human being. With love, we can conquered all the difficulties.
But love for nature is not often mentioned by the public. We love the beautiful scenery and the lovely animals, so we have the desire to protect the enviroment, for the purpose of keeping clean scenery and make sure the animals won't disappear. Animals and human being are part of nature. We live and die together.
Love is not just around families and friends, but we also need to care for nature.''')
    print time.time()-time1
