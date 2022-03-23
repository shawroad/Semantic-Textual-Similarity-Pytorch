"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-03-23
"""
import torch
import json
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def load_data(path):
    sentences = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            sentences.append(line['origin'])
            sentences.append(line['entailment'])
            sentences.append(line['contradiction'])
            # 注意 后续不能对这些数据打乱,不然关系就乱了  想shuffle可以提前shuffle
            # 还有一点 就是batch_size必须是3的倍数
    return sentences


def load_test_data(path):
    sent1, sent2, label = [], [], []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            sent1.append(line[0])
            sent2.append(line[1])
            label.append(int(line[2]))
    return sent1, sent2, label


class TrainDataSet(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = data
        self.data_set = []
        for d in self.data:
            input_ids, attention_mask = self.convert_feature(d)
            self.data_set.append({'input_ids': input_ids, 'attention_mask': attention_mask})

    def convert_feature(self, sample):
        # 将文本转为id序列
        input_ids = self.tokenizer.encode(sample)
        attention_mask = [1] * len(input_ids)
        return input_ids, attention_mask

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        instance = self.data_set[idx]
        return instance


def collate_train_func(batch_data):
    '''
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    :param batch_data: batch数据
    :return:
    '''
    # max_len = max[len(d['input_ids']) for d in batch_data]
    input_ids_list, attention_mask_list = [], []
    for instance in batch_data:
        input_ids_temp = instance['input_ids']
        attention_mask_temp = instance['attention_mask']
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask_temp, dtype=torch.long))

        # 将input_ids_temp和token_type_ids_temp添加到对应的list中
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "attention_mask_ids": pad_sequence(attention_mask_list, batch_first=True, padding_value=0)}


if __name__ == '__main__':
    path = './data/train.txt'
    load_data(path)


