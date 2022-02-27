"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-02-23
"""
import torch
import random
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def load_data(path):
    sent1, sent2, label = [], [], []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            sent1.append(line[0])
            sent2.append(line[1])
            label.append(int(line[2]))
    df = pd.DataFrame({'sent1': sent1, 'sent2': sent2, 'label': label})
    return df


class SentDataSet(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = data    # pd.DataFrame({'sent1': sent1, 'sent2': sent2, 'label': label})
        self.sent1 = self.data['sent1']
        self.sent2 = self.data['sent2']
        self.label = self.data['label']

    def __len__(self):
        return len(self.sent1)

    def __getitem__(self, idx):
        s1_input_ids = self.tokenizer.encode(self.sent1[idx])
        s2_input_ids = self.tokenizer.encode(self.sent2[idx])
        return {'s1_input_ids': s1_input_ids, 's2_input_ids': s2_input_ids, 'label': self.label[idx]}


def pad_to_maxlen(input_ids, max_len, pad_value=0):
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
    return input_ids


def collate_func(batch_data):
    '''
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    :param batch_data: batch数据
    :return:
    '''
    s1_max_len = max([len(d['s1_input_ids']) for d in batch_data])
    s2_max_len = max([len(d['s2_input_ids']) for d in batch_data])

    s1_input_ids_list, s2_input_ids_list, label_list = [], [], []
    for item in batch_data:
        s1_input_ids_list.append(pad_to_maxlen(item['s1_input_ids'], max_len=s1_max_len))
        s2_input_ids_list.append(pad_to_maxlen(item['s2_input_ids'], max_len=s2_max_len))
        label_list.append(item['label'])
    all_s1_input_ids = torch.tensor(s1_input_ids_list, dtype=torch.long)
    all_s2_input_ids = torch.tensor(s2_input_ids_list, dtype=torch.long)
    all_labels_id = torch.tensor(label_list, dtype=torch.long)   # 分类这里为torch.long  回归这里为torch.float
    return all_s1_input_ids, all_s2_input_ids, all_labels_id
