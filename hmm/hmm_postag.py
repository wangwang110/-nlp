"""
hmm用于词性标注，有标注语料，使用极大似然估计学习参数，词性是隐状态序列，词是观测序列
三个基本问题：
1.学习
2.解码
3.预测
"""
import numpy as np

# 1. 学习，统计得到参数，A，B，pi

# 统计所有的单词和词性
word2id = {}
tag2id = {}
with open("./postagging_traindata.txt") as f:
    for line in f.readlines():
        word, tag = line.strip().split("/")
        if word not in word2id:
            word2id[word] = len(word2id)
        if tag not in tag2id:
            tag2id[tag] = len(tag2id)

# 统计得到，初始状态，转移矩阵，发射矩阵
m = len(word2id)
n = len(tag2id)
pi = np.zeros([n])
# 第一个位置每一个隐状态的概率pi
A = np.zeros([n, n])
B = np.zeros([n, m])

pre_id = tag2id["."]
with open("./postagging_traindata.txt") as f:
    for line in f.readlines():
        word, tag = line.strip().split("/")
        word_id = word2id[word]
        tag_id = tag2id[tag]
        if pre_id == tag2id["."]:
            pi[tag_id] += 1
            A[pre_id][tag_id] += 1
            B[tag_id][word_id] += 1
            pre_id = tag_id
        else:
            A[pre_id][tag_id] += 1
            B[tag_id][word_id] += 1
            pre_id = tag_id

pi = pi / np.sum(pi)
A = A / np.sum(A, 1)[:, None]
B = B / np.sum(B, 1)[:, None]
###


def log(x):
    if x == 0:
        return np.log(x + 1e-4)
    else:
        return np.log(x)


# 2. 解码，给定观测序列得到最有可能的隐状态序列，也就是词性（维特比解码），时间复杂度O(n**2*T)
seq = "I like playing the football"
seq_ids = [word2id[word] for word in seq.split()]
T = len(seq_ids)

# 动态规划的过程就是填表的过程
# 需要回溯,得到词性
ptr = np.zeros((n, T), dtype=int)  # 存放下标

viterbi_x = np.zeros((n, T), dtype=float)
for i in range(n):
    viterbi_x[i][0] = log(pi[i]) + log(B[i][seq_ids[0]])

# 填表，每个位置保留转移过来最大的
for i in range(n):
    for j in range(1, T):
        viterbi_x[i][j] = float("-inf")
        for t in range(n):
            score = viterbi_x[t][j - 1] + log(A[t][i]) + log(B[i][seq_ids[j]])
            if score > viterbi_x[i][j]:
                viterbi_x[i][j] = score
                ptr[i][j] = t
                # 记录从哪个隐状态转移过来的

# 回溯过程
best_seq = [0] * T
best_seq[T - 1] = np.argmax(viterbi_x[:, T - 1])

for i in range(T - 2, -1, -1):  # T-1 -> 0
    best_seq[i] = ptr[best_seq[i + 1]][i + 1]

id2tag = {index: word for (word, index) in tag2id.items()}
for i in range(len(best_seq)):
    print(id2tag[best_seq[i]])
