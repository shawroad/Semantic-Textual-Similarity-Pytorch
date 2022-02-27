"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-02-23
"""
import argparse


def set_args():
    # ../data/ATEC/ATEC.train.data
    # ../data/ATEC/ATEC.test.data
    parser = argparse.ArgumentParser('--使用transformers实现sentence_bert')
    parser.add_argument('--train_data_path', default='../data/ATEC/ATEC.train.data', type=str, help='训练数据集')
    parser.add_argument('--dev_data_path', default='../data/ATEC/ATEC.test.data', type=str, help='测试数据集')
    parser.add_argument('--bert_pretrain_path', default='./roberta_pretrain', type=str, help='预训练模型路径')
    parser.add_argument('--train_batch_size', default=32, type=int, help='训练批次的大小')
    parser.add_argument('--dev_batch_size', default=32, type=int, help='训练批次的大小')
    parser.add_argument('--output_dir', default='./output', type=str, help='模型输出目录')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度积聚的大小')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='训练几轮')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='学习率大小')
    return parser.parse_args()
