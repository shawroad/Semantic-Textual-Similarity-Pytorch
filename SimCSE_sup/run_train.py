"""
@file   : run_train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-03-23
"""
import os
import random
import torch
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import nn
from model import Model
from config import set_args
import torch.nn.functional as F
from einops import repeat, rearrange
from torch.utils.data import Dataset, DataLoader
from utils import l2_normalize, compute_corrcoef, compute_pearsonr
from data_helper import TrainDataSet, collate_train_func, load_test_data, load_data
from transformers import BertTokenizer, get_linear_schedule_with_warmup, AdamW


def pad_to_maxlen(input_ids, max_len, pad_value=0):
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
    return input_ids


def get_sent_id_tensor(s_list):
    input_ids, attention_mask, token_type_ids, labels = [], [], [], []
    max_len = max([len(_)+2 for _ in s_list])   # 这样写不太合适 后期想办法改一下
    for s in s_list:
        inputs = tokenizer.encode_plus(text=s, text_pair=None, add_special_tokens=True, return_token_type_ids=True)
        input_ids.append(pad_to_maxlen(inputs['input_ids'], max_len=max_len))
        attention_mask.append(pad_to_maxlen(inputs['attention_mask'], max_len=max_len))
        token_type_ids.append(pad_to_maxlen(inputs['token_type_ids'], max_len=max_len))
    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(attention_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(token_type_ids, dtype=torch.long)
    return all_input_ids, all_input_mask, all_segment_ids


def evaluate():
    sent1, sent2, label = load_test_data(args.test_data)
    all_a_vecs = []
    all_b_vecs = []
    all_labels = []
    model.eval()
    for s1, s2, lab in tqdm(zip(sent1, sent2, label)):
        input_ids, input_mask, segment_ids = get_sent_id_tensor([s1, s2])
        lab = torch.tensor([lab], dtype=torch.float)
        if torch.cuda.is_available():
            input_ids, input_mask, segment_ids = input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()
            lab = lab.cuda()

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=input_mask, encoder_type='cls')

        all_a_vecs.append(output[0].cpu().numpy())
        all_b_vecs.append(output[1].cpu().numpy())
        all_labels.extend(lab.cpu().numpy())

    all_a_vecs = np.array(all_a_vecs)
    all_b_vecs = np.array(all_b_vecs)
    all_labels = np.array(all_labels)

    a_vecs = l2_normalize(all_a_vecs)
    b_vecs = l2_normalize(all_b_vecs)
    sims = (a_vecs * b_vecs).sum(axis=1)
    corrcoef = compute_corrcoef(all_labels, sims)
    pearsonr = compute_pearsonr(all_labels, sims)
    return corrcoef, pearsonr


def compute_loss(y_pred):
    device = y_pred.device
    y_true = torch.arange(y_pred.shape[0], device=device)
    # print(y_true)   # [0, 1, 2, 3, 4, 5, 6, 7, 8]

    use_row = torch.where((y_true + 1) % 3 != 0)[0]   # 有用的行 即: 难负例除外
    # print(use_row)   # [0, 1, 3, 4, 6, 7]

    y_true = (use_row - use_row % 3 * 2) + 1
    # print(y_true)   # [1, 0, 4, 3, 7, 6]  有用行对应的label

    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    # print(sim.size())    # torch.Size([27, 27])

    # 选取有效的行
    sim = torch.index_select(sim, 0, use_row)
    # print(sim.size())    # torch.Size([18, 27])

    # 相似度矩阵除以温度系数
    sim = sim / 0.05

    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return loss


if __name__ == '__main__':
    args = set_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_texts = load_data(args.train_data)

    # 2. 构建一个数据加载器
    tokenizer = BertTokenizer.from_pretrained('./roberta_pretrain/vocab.txt')
    train_data = TrainDataSet(train_texts, tokenizer)
    train_data_loader = DataLoader(train_data, batch_size=args.train_batch_size, collate_fn=collate_train_func)
    total_steps = int(len(train_data_loader) * args.num_train_epochs / args.gradient_accumulation_steps)

    print("总训练步数为:{}".format(total_steps))
    model = Model()
    if torch.cuda.is_available():
        model.cuda()

    # 获取模型所有参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # 设置优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)

    for epoch in range(args.num_train_epochs):
        model.train()
        temp_loss = 0
        count = 0
        for step, batch in enumerate(train_data_loader):
            count += 1
            start_time = time.time()
            input_ids = batch["input_ids"]  # torch.Size([6, 22])
            attention_mask_ids = batch['attention_mask_ids']

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask_ids = attention_mask_ids.cuda()

            outputs = model(input_ids, attention_mask_ids, encoder_type='cls')
            loss = compute_loss(outputs)
            temp_loss += loss.item()

            # 将损失值放到Iter中，方便观察
            ss = 'Epoch:{}, Step:{}, Loss:{:10f}, Time_cost:{:10f}'.format(epoch, step, loss, time.time() - start_time)
            print(ss)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # 损失进行回传
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # 当训练步数整除累积步数时，进行参数优化
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        train_loss = temp_loss / count
        corr, pears = evaluate()
        s = 'Epoch:{} | cur_epoch_average_loss:{:10f} |spearmanr: {:10f} | pearsonr: {:10f}'.format(epoch, train_loss, corr, pears)
        logs_path = os.path.join(args.output_dir, 'logs.txt')
        with open(logs_path, 'a+') as f:
            s += '\n'
            f.write(s)

        # 每个epoch进行完，则保存模型
        output_dir = os.path.join(args.output_dir, "Epoch-{}.bin".format(epoch))
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), output_dir)
        # 清空cuda缓存
        torch.cuda.empty_cache()
