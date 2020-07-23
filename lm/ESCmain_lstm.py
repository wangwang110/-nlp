# encoding=utf-8
from __future__ import division
import math
import os
import nltk
import pickle
import argparse
import numpy as np
import distance
import tensorflow as tf
import lcs
import time
from nltk.corpus import wordnet
from data_utils import build_one


def getscores(sents, args, sess):
    """
    :param sents:
    :param args:
    :return:
    """
    all_candidate_scores = []

    if os.path.isfile(args.word_dict_path):
        word_dict = pickle.load(open(args.word_dict_path, "rb"))
    else:
        print("load word_dict error !!!")

    test_left, test_right = build_one(sents, word_dict, args.max_len)

    # with tf.Session() as sess:
    #     saver = tf.train.import_meta_graph("./lm/bi-rnn-4751.meta")
    #     saver.restore(sess, tf.train.latest_checkpoint("./lm"))
    #     # 五个模型要用不同的存储模型的文件夹，因为文件夹下有自动命名的相同文件checkpoint
    #
    #     graph = tf.get_default_graph()
    #     x = graph.get_tensor_by_name("x:0")
    #     x_r = graph.get_tensor_by_name("x_r:0")
    #
    #     keep_prob = graph.get_tensor_by_name("keep_prob:0")
    #     seq_len = graph.get_tensor_by_name("seq_len:0")
    #     # 要改正的句子的原有的类别，不一定正确
    #
    #     logits = graph.get_tensor_by_name("output/logits/BiasAdd:0")
    #     probs = tf.nn.softmax(logits, axis=2, name="probs")

    feed_dict = {x: test_left, x_r: test_right, keep_prob: 1.0}
    predicts = sess.run(probs, feed_dict=feed_dict)

    num = len(sents)
    for i in range(num):
        pos_scores = 1.0
        real_len = len(sents[i].split(" "))
        li_seq = test_left[i][1:]
        for j in range(real_len):
            pos_scores *= predicts[i][j][li_seq[j]]
        all_candidate_scores.append(math.log(pos_scores))
    return all_candidate_scores


def get_lexicon_list(path):
    word_list = []
    file_object = open(path)
    for line in file_object:
        line = line.strip().replace("\n", "").replace("\r", "")
        word_list.append(line)
    return word_list


def splitSentence(paragraph):
    # nltk自带的分句文件，以句号分
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)
    return sentences


def get_candidates(word):
    # 获取一个词的候选集列表，根据分值
    s = open('./candidates.pkl', 'rb')
    candidates = pickle.load(s)
    s.close()

    candidates_words = []

    if word not in candidates.keys():
        # 词不在离线计算的文件中，重新计算加入
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

        if temp[0][0] != word[0]:
            # 首字母相同
            score = 0.5 * score

        if score > 2:
            candidates_words.append((temp[0], score))
    return candidates_words


def replace(words, index, args, sess, isTrue=False):
    original_word = words[index]
    original_sent = " ".join(words)
    candidates_words = get_candidates(original_word)

    if len(candidates_words) == 0:
        return original_word

    candidate_sents = [original_sent]
    li_words = []
    li_score = []

    for word, word_score in candidates_words:
        words[index] = word
        li_words.append(word)
        li_score.append(word_score)
        sent = " ".join(words)
        candidate_sents.append(sent)

    probs = getscores(candidate_sents, args, sess)
    original_prob = probs[0]
    logprobs = probs[1:]

    candidate_probs = {}
    for k, v, t in zip(li_words, logprobs, li_score):
        print(v, t)
        candidate_probs[k] = v + 10*t
        # 加上 t

    result = max(candidate_probs.items(), key=lambda s: s[1])

    # #(key,value)
    #     # if result[1]-original_score > original_score *0.3:
    #     #     return  result[0]
    #     # else:
    #     #     return  original_word
    return result[0]


def process_txt(args,sess):
    punctuations = [',', '(', ')', '.', '!', '"', '?', 'st', 'ed']

    dictionary_object = open('dictionary_youdao_1.txt')
    dictionary = []
    for line in dictionary_object:
        line = line.strip().replace("\n", "").replace("\r", "")
        dictionary.append(line)
    print(len(dictionary))

    # vocab = pickle.load(open("/container_data/birnn-language-model-tf-master/ptb_data/vocab.bin", "rb"))
    # dictionary.extend(vocab)
    # dictionary = list(set(dictionary))
    # print(len(dictionary))

    file_object = open(args.path)
    try:
        all_the_text = file_object.read()
    finally:
        file_object.close()

    list_sentence = splitSentence(all_the_text)
    result_file = open('result_false.txt', 'w')
    wrong_sample = open("./lm.error", "w")

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
        # 注意分词工具会根据空格和标点分

        list_sentence_word = list_sentence_word[:-1]
        # 去掉末尾的.号，不去也没什么问题，怕会影响概率计算
        i = 0
        for temp_word in list_sentence_word:
            temp_word_lemma = temp_word
            if temp_word_lemma not in punctuations and temp_word_lemma not in dictionary:
                # if len(list_sentence_word)+2 > args.max_len:
                start = max(0, i-10)
                end = min(i+10, len(list_sentence_word))
                tmp_sentence_word = list_sentence_word[start:end]
                # 不改变 list_sentence_word
                correct_word = replace(tmp_sentence_word, i-start, args, sess, False)
                # else:
                #     correct_word = replace(list_sentence_word, i, args, sess, False)

                if temp_word != correct_word:
                    error_false[temp_word] = correct_word
                    if correct_word != list_true_sentence_word[i]:
                        print(correct_word, list_true_sentence_word[i])
                        wrong_sample.write(" ".join(list_true_sentence_word) + "\n")
                        wrong_sample.write(temp_word + "," + correct_word + "," + list_true_sentence_word[i] + "\n")
            i = i + 1

        if len(error_false.keys()) != 0:
            predictTrue += 1
            original_sentence_word = nltk.word_tokenize(sentence)
            # reset一下,list_sentence_word 内容变了，在replace的过程当中
            sentence_len = len(original_sentence_word)

            for index in range(sentence_len):
                temp = original_sentence_word[index]
                if temp in error_false:
                    original_sentence_word[index] = error_false[temp]

            if ' '.join(original_sentence_word[:-1]) + "." == list_sentence[j - 1]:
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
    file_result.close()
    result_file.close()
    wrong_sample.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=116, help="max_sequence_length")
    parser.add_argument("--path", type=str, default='./test_file_04.txt', help="spell_correct")

    # 由训练集文件得到
    parser.add_argument("--word_dict_path", type=str,
                        default="/container_data/birnn-language-model-tf-master/ptb_data/word2id.bin",
                        help="train file word2id")

    parser.add_argument("--embedding_matrix_path", type=str,
                        default="/container_data/birnn-language-model-tf-master/ptb_data/embedding.npy",
                        help="embedding matrix")

    # 由词向量文件得到，不会改变
    parser.add_argument("--embedding_file_path", type=str,
                        default="/container_data/birnn-language-model-tf-master/ptb_data/word2vec.bin",
                        help="embedding_file_path")

    args = parser.parse_args()
    start_time = time.time()

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("/container_data/birnn-language-model-tf-master/lm1/bi-rnn-534301.meta")
        saver.restore(sess, tf.train.latest_checkpoint("/container_data/birnn-language-model-tf-master/lm1"))
        # 五个模型要用不同的存储模型的文件夹，因为文件夹下有自动命名的相同文件checkpoint

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        x_r = graph.get_tensor_by_name("x_r:0")

        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        seq_len = graph.get_tensor_by_name("seq_len:0")
        # 要改正的句子的原有的类别，不一定正确

        logits = graph.get_tensor_by_name("output/logits/BiasAdd:0")
        probs = tf.nn.softmax(logits, axis=2, name="probs")
        end_time = time.time()
        print("load model time：", end_time - start_time)
        process_txt(args, sess)
        print("all time", time.time() - start_time)
