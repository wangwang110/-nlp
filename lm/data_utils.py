import collections
import numpy as np
import pickle
import random
import copy


def build_word_dict(filename, path):
    with open(filename, "r") as f:
        words = f.read().replace("\n", "").split()

    word_counter = collections.Counter(words).most_common(40000)
    word_dict = dict()
    word_dict["<pad>"] = 0
    word_dict["<bos>"] = 1
    word_dict["<eos>"] = 2
    word_dict["<unk>"] = 3
    for word, _ in word_counter:
        word_dict[word] = len(word_dict)
    pickle.dump(word_dict, open(path, "wb"))
    return word_dict


def load_word_vector(w2v_file, word2id, embedding_dim, path):
    # load_word_vector(args.embedding_file_path, word_dict, args.embedding_size, args.embedding_matrix_path)
    # load external word-vector file
    # print(w2v_file)
    # fp = open(w2v_file)
    # word_vector = dict()
    # for line_ in fp:
    #     line = line_.split(' ')
    #     if len(line) != embedding_dim + 1:
    #         print("invalid word embedding：{}".format(line[0]))
    #         continue
    #     else:
    #         word_vector[line[0]] = [float(v) for v in line[1:]]

    words_num = len(word2id.keys())
    word_vector = pickle.load(open(w2v_file, "rb"))
    li_wordvector = []
    for i in range(words_num):
        li_wordvector.append(np.random.uniform(-0.01, 0.01, (embedding_dim,)))

    for k in word2id.keys():
        if k in word_vector.keys():
            li_wordvector[word2id[k]] = word_vector[k]

    li_wordvector[0] = [0.] * embedding_dim  # for <pad>

    w2v = np.asarray(li_wordvector, dtype=np.float32)
    np.save(path, w2v)
    # 与测试集无关
    # i=0
    # for k in word_vector.keys():
    #     if k not in word2id.keys():
    #         i += 1
    #         word2id[k] = len(word2id)
    #         w2v = np.row_stack((w2v, word_vector[k]))
    #         print(i)
    # # 本来加上这一段可以把词向量文件中的词加入词向量矩阵中，这样如果测试集中出现的词可能在词向量文件中，但是不再训练集中就可以进行处理。但是太慢了
    return w2v


def build_dataset(filename, word_dict, max_len=None):
    with open(filename, "r") as f:
        lines = f.readlines()
        if max_len == None:
            random.shuffle(lines)
            lines = lines[:10000000]

    data = list(map(lambda s: s.strip().split(), lines))
    if max_len == None:
        max_document_len = max([len(s) for s in data]) + 2
    else:
        max_document_len = max_len
        texts = []
        for s in data:
            if len(s)+2 <= max_document_len:
                texts.append(s)
            else:
                continue
        data = texts

    data = list(map(lambda s: ["<bos>"] + s + ["<eos>"], data))
    data = list(map(lambda s: [word_dict.get(w, word_dict["<unk>"]) for w in s], data))
    data_right = copy.deepcopy(data)
    data = list(map(lambda d: d + (max_document_len - len(d)) * [word_dict["<pad>"]], data))
    for tmp in data_right:
        tmp.reverse()
    data_right = list(map(lambda d: d + (max_document_len - len(d)) * [word_dict["<pad>"]], data_right))

    return [data, data_right]


def build_one(texts, word_dict, max_len=None):
    # lines = texts
    data = list(map(lambda s: s.strip().split(), texts))
    # max_document_len = max_len
    # data = []
    # for s in li_tmp:
    #     if len(s) + 2 <= max_document_len:
    #         data.append(s)
    #     else:
    #         continue

    data = list(map(lambda s: ["<bos>"] + s + ["<eos>"], data))
    data = list(map(lambda s: [word_dict.get(w, word_dict["<unk>"]) for w in s], data))
    data_right = copy.deepcopy(data)
    data = list(map(lambda d: d + (max_len - len(d)) * [word_dict["<pad>"]], data))
    for tmp in data_right:
        tmp.reverse()
    data_right = list(map(lambda d: d + (max_len - len(d)) * [word_dict["<pad>"]], data_right))

    return [data, data_right]


def batch_iter(inputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index]
