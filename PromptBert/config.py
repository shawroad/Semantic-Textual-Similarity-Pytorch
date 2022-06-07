"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-06-07
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser('--PromptBert')
    parser.add_argument('--train_data_path', default='../data/ATEC/ATEC.train.data', type=str, help='训练数据集')
    parser.add_argument('--dev_data_path', default='../data/ATEC/ATEC.test.data', type=str, help='测试数据集')
    parser.add_argument('--max_len', default=48, type=int, help='句子的最大长度')
    parser.add_argument('--train_batch_size', default=4, type=int, help='训练批次的大小')
    parser.add_argument('--dev_batch_size', default=4, type=int, help='训练批次的大小')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='训练几轮')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='学习率大小')
    parser.add_argument('--bert_pretrain_path', default='./roberta_pretrain', type=str, help='预训练模型路径')
    parser.add_argument('--output_dir', default='./output', type=str, help='模型输出目录')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度积聚的大小')
    return parser.parse_args()