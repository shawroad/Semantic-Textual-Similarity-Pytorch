"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-08-23
"""
import argparse


def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()

    # ./data/ATEC/ATEC.train.data
    # ./data/BQ/BQ.train.data
    # ./data/LCQMC/LCQMC.train.data
    # ./data/PAWSX/PAWSX.train.data
    # ./data/STS-B/STS-B.train.data 
    parser.add_argument('--train_data', default='../data/LCQMC/LCQMC.train.data', type=str, help='训练数据集')
    parser.add_argument('--test_data', default='../data/LCQMC/LCQMC.test.data', type=str, help='测试数据集')
    parser.add_argument('--train_batch_size', type=int, default=32, help='训练批次大小')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='总共训练几轮')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Adam优化器的epsilon值')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='warm up概率，即训练总步长的百分之多少，进行warm up')
    parser.add_argument('--learning_rate', default=9e-6, type=float, help='模型训练时的学习率')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--output_dir', default='./outputs', type=str, help='模型输出文件夹')
    return parser.parse_args()
