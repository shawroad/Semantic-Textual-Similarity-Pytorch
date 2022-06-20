"""
@file   : inference.py.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-06-20
"""
import gensim
import scipy
import scipy.stats
import jieba
import numpy as np
from tqdm import tqdm


def load_test_data(path):
    sent1_list, sent2_list, label_list = [], [], []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            sent1, sent2, label = line.split('\t')
            sent1_list.append(sent1)
            sent2_list.append(sent2)
            label_list.append(int(label))
    return sent1_list, sent2_list, label_list


def get_sent_vec(sent):
    sent_vocab = jieba.lcut(sent)

    vec_list = []
    for v in sent_vocab:
        try:
            vec = model.wv[v]
            vec_list.append(vec.tolist())
        except:
            continue
    if len(vec_list) == 0:
        return None
    vec_arr = np.array(vec_list)
    max_pooling_vec = vec_arr.max(0)
    return max_pooling_vec


def cosine_sim(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))

    if a.ndim == 1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)

    elif a.ndim == 2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)

    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))

    similiarity = np.dot(a, b.T) / (a_norm * b_norm.T)  # 计算相似度 [-1,1]
    return similiarity


def compute_spearman(x, y):
    """
    斯皮尔曼Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation


def compute_pearsonr(x, y):
    '''
    皮尔逊pearsonr相关系数
    '''
    return scipy.stats.pearsonr(x, y)[0]


if __name__ == '__main__':
    # 加载测试集
    # path = './data/ATEC/ATEC.test.data'
    # path = './data/BQ/BQ.test.data'
    # path = './data/LCQMC/LCQMC.test.data'
    # path = './data/PAWSX/PAWSX.test.data'
    path = './data/STS-B/STS-B.test.data'
    sent1_list, sent2_list, label_list = load_test_data(path)

    # 加载模型
    model = gensim.models.Word2Vec.load('word2vec_model.model')

    score_list = []
    for sent1, sent2 in tqdm(zip(sent1_list, sent2_list)):
        sent1_vec = get_sent_vec(sent1)
        sent2_vec = get_sent_vec(sent2)
        if sent1_vec is not None and sent2_vec is not None:
            score = cosine_sim(sent1_vec, sent2_vec)
            score_list.append(score)
        else:
            score_list.append(0)  # 得不到向量的认为不相关

    # 评价
    spearman_coff = compute_spearman(label_list, score_list)
    pearsonr_coff = compute_pearsonr(label_list, score_list)
    print('斯皮尔曼系数:', spearman_coff)
    print('皮尔逊系数:', pearsonr_coff)
