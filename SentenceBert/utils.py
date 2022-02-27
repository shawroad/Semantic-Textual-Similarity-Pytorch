"""
@file   : utils.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-02-23
"""
import scipy
import numpy as np
import scipy.stats


def l2_normalize(vecs):
    """标准化
    """
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation


def compute_pearsonr(x, y):
    # 输出:(r, p)
    # r:相关系数[-1，1]之间
    # p:相关系数显著性
    # 所有下面的数据选第零位
    return scipy.stats.pearsonr(x, y)[0]