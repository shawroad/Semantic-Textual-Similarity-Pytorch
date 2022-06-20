"""
@file   : run_train_word2vec.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-06-20
"""
import gensim
import jieba
from tqdm import tqdm


def load_data(path):
    all_corpus = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            try:
                all_corpus.append(line[0])
                all_corpus.append(line[1])
            except:
                continue
    return all_corpus


def load_stopword():
    temp = []
    path = './data/stopword.txt'
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            temp.append(line)
    temp.append(' ')
    return temp


if __name__ == '__main__':
    # 1. 加载数据
    # train_data_path = './data/ATEC/ATEC.train.data'
    # train_data_path = './data/BQ/BQ.train.data'
    # train_data_path = './data/LCQMC/LCQMC.train.data'
    # train_data_path = './data/PAWSX/PAWSX.train.data'
    train_data_path = './data//STS-B/STS-B.train.data'
    corpus = load_data(train_data_path)

    # 2. 加载停用词表
    stopwords = load_stopword()

    # 3. 切词并去除停用词
    train_corpus = []
    for sent in tqdm(corpus):
        # sent = [w for w in jieba.lcut(sent) if w not in stopwords]   # 加上停用词效果不太好
        sent = [w for w in jieba.lcut(sent)]
        train_corpus.append(sent)
    # 训练
    model = gensim.models.Word2Vec(sentences=train_corpus, min_count=2, window=5, vector_size=128, seed=43, workers=3)
    model.save('word2vec_model.model')
