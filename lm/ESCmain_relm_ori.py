# encoding=utf-8
from __future__ import division
import math
import nltk
import pickle
import numpy as np
import distance
import lcs
import time
# from pattern.en import lemma
from nltk.corpus import wordnet
import time


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


def get_log_prob(words, dict_list, vocab_size, lexcion_size):
    probs = getprobs(words, dict_list, vocab_size, lexcion_size)
    return math.log(probs)


def sentence_log_prob(words, dict_ngram, vocab_size, lexcion_size, n_gram=3):
    ##获取一个句子中，所有的ngram概率返回一个list
    logprobs = []
    if (len(words) <= n_gram):
        logprobs.append(get_log_prob(words, dict_ngram, vocab_size, lexcion_size))
    else:
        wordngrams = nltk.ngrams(words, n_gram)
        for wordTrigram in wordngrams:
            logprobs.append(get_log_prob(wordTrigram, dict_ngram, vocab_size, lexcion_size))
    return logprobs


def get_lexicon_list(path):
    word_list = []
    file_object = open(path)
    for line in file_object:
        line = line.strip().replace("\n", "").replace("\r", "")
        word_list.append(line)
    return word_list


def splitSentence(paragraph):
    ## nltk自带的分句文件，以句号分
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)
    return sentences


def get_candidates(word, candidates):
    ## 获取一个词的候选集列表，根据分值

    candidates_words = []

    if word not in candidates.keys():
        ## 词不在离线计算的文件中，重新计算加入

        edits1 = distance.edits1(word)
        edits2 = distance.edits2(word)

        temp = open('f2true.pkl', 'rb')
        f2true = pickle.load(temp)
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
        score = temp[1] * 2 + 0.9 * temp[2] + 0.5 * temp[3]

        if (temp[0][0] != word[0]):
            ##首字母相同
            score = 0.5 * score

        if (score > 2):
            candidates_words.append((temp[0], score))

    return candidates_words


def replace(words, index, dict_list, vocab_size, lexcion_size, candidates, n_gram=3):

    original_word = words[index]
    candidates_words = get_candidates(words[index], candidates)

    if (len(candidates_words) == 0):
        ##没有候选单词返回原有的单词，不改
        return original_word

    candidate_probs = {}
    start = max(0, index - n_gram + 1)
    end = min(len(words) - 1, index + n_gram)
    ##避免越界

    for word in candidates_words:
        words[index] = word[0]
        ##替换原词为候选词，计算概率
        temp = words[start:end]
        logprobs = sentence_log_prob(temp, dict_list, vocab_size, lexcion_size)
        candidate_probs[word[0]] = np.mean(logprobs) + word[1]
        # print (np.mean(logprobs)+1)*word[1]
        ###求和也可以

    result = max(candidate_probs.items(), key=lambda s: s[1])
    # (key,value)

    return result[0]


def loadfile(ngram=3):
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


def load_all_file(ngram):
    dict_list = loadfile(ngram)
    vocab_size, lexcion_size = open("size.txt", "r").read().strip().split(" ")
    return [dict_list, vocab_size, lexcion_size]


def process_txt(path):
    start = time.time()
    ngram = 5
    dict_list, vocab_size, lexcion_size = load_all_file(ngram)

    s = open('candidates.pkl', 'rb')
    candidates = pickle.load(s)
    s.close()

    print("load model time", time.time() - start)

    punctuations = [',', '(', ')', '.', '!', '"', '?', 'st', 'ed']

    dictionary_object = open('dictionary_youdao_1.txt')
    dictionary = []
    for line in dictionary_object:
        line = line.strip().replace("\n", "").replace("\r", "")
        dictionary.append(line)

    file_object = open(path)
    try:
        all_the_text = file_object.read()
    finally:
        file_object.close()

    list_sentence = splitSentence(all_the_text)
    result_file = open('result_false.txt', 'w')

    sentences_num = len(list_sentence)
    predictTrue = 0
    goldTrue = sentences_num / 2
    tp = 0

    for j in range(1, sentences_num, 2):
        print(j)
        error_false = {}
        sentence = list_sentence[j]
        list_true_sentence_word = nltk.word_tokenize(list_sentence[j - 1])
        list_sentence_word = nltk.word_tokenize(sentence)
        list_sentence_word = list_sentence_word[:-1]

        i = 0
        for temp_word in list_sentence_word:
            temp_word_lemma = temp_word
            # result_lenght=len(wordnet.synsets(temp_word_lemma))
            ## 假词错误检测及改正               
            if temp_word_lemma not in punctuations and temp_word_lemma not in dictionary:
                correct_word = replace(list_sentence_word, i, dict_list, vocab_size, lexcion_size, candidates)
                if (temp_word != correct_word):
                    error_false[temp_word] = correct_word
                    if (correct_word != list_true_sentence_word[i]):
                        print(correct_word, list_true_sentence_word[i])
            i = i + 1

        if (len(error_false.keys()) != 0):
            predictTrue += 1
            original_sentence_word = nltk.word_tokenize(sentence)
            sentence_len = len(original_sentence_word)

            for index in range(sentence_len):
                temp = original_sentence_word[index]
                if temp in error_false:
                    original_sentence_word[index] = error_false[temp]

            if (' '.join(original_sentence_word[:-1]) + "." == list_sentence[j - 1]):
                tp += 1

        result_file.writelines(list_sentence[j])
        result_file.writelines("\n")
        result_file.writelines(list_sentence[j - 1])

        result_file.write('\nfalse word error:')
        result_file.write(str(error_false))
        result_file.writelines('\n')
        result_file.flush()

    P = (1.0 * tp) / predictTrue
    R = (1.0 * tp) / goldTrue
    file_result = open("F1_result.txt", 'a')
    print('p:', P)
    file_result.write("P:%f" % P)
    print('R:', R)
    file_result.write("R:%f" % R)
    print('the F1-score is:', (2 * P * R) / (P + R))
    file_result.write("F1:%f" % ((2 * P * R) / (P + R)))
    print("load handle time", time.time() - start)
    file_result.close()

    result_file.close()
    s.close()


if __name__ == '__main__':
    path = './test_file_04.txt'
    process_txt(path)
