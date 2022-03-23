"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-03-23
"""
import argparse


def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='./data/train.txt', type=str, help='训练数据集')
    parser.add_argument('--test_data', default='./data/test_convert.txt', type=str, help='测试数据集')
    parser.add_argument('--train_batch_size', type=int, default=27, help='训练批次大小')
    parser.add_argument('--num_train_epochs', type=int, default=5, help='总共训练几轮')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Adam优化器的epsilon值')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='warm up概率，即训练总步长的百分之多少，进行warm up')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='模型训练时的学习率')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--output_dir', default='./outputs', type=str, help='模型输出文件夹')
    return parser.parse_args()