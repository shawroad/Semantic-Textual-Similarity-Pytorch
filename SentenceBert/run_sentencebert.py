"""
@file   : run_sentencebert.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-02-23
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from config import set_args
from model import Model
from torch.utils.data import DataLoader
from utils import compute_corrcoef, l2_normalize, compute_pearsonr
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.models.bert import BertTokenizer
from data_helper import load_data, SentDataSet, collate_func


def evaluate(model):
    model.eval()
    # 语料向量化
    all_a_vecs, all_b_vecs = [], []
    all_labels = []
    for step, batch in tqdm(enumerate(val_dataloader)):
        if torch.cuda.is_available():
            batch = (t.cuda() for t in batch)
        s1_input_ids, s2_input_ids, label_id = batch
        if torch.cuda.is_available():
            s1_input_ids, s2_input_ids, label_id = s1_input_ids.cuda(), s2_input_ids.cuda(), label_id.cuda()
        with torch.no_grad():
            s1_embeddings = model(s1_input_ids, encoder_type='cls')
            s2_embeddings = model(s2_input_ids, encoder_type='cls')
            s1_embeddings = s1_embeddings.cpu().numpy()
            s2_embeddings = s2_embeddings.cpu().numpy()
            label_id = label_id.cpu().numpy()

            all_a_vecs.extend(s1_embeddings)
            all_b_vecs.extend(s2_embeddings)
            all_labels.extend(label_id)

    all_a_vecs = np.array(all_a_vecs)
    all_b_vecs = np.array(all_b_vecs)
    all_labels = np.array(all_labels)

    a_vecs = l2_normalize(all_a_vecs)
    b_vecs = l2_normalize(all_b_vecs)
    sims = (a_vecs * b_vecs).sum(axis=1)
    corrcoef = compute_corrcoef(all_labels, sims)
    pearsonr = compute_pearsonr(all_labels, sims)
    return corrcoef, pearsonr


def calc_loss(s1_vec, s2_vec, true_label):
    loss_fct = nn.MSELoss()
    output = torch.cosine_similarity(s1_vec, s2_vec)
    loss = loss_fct(output, true_label)
    return loss


if __name__ == '__main__':
    args = set_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain_path)

    train_df = load_data(args.train_data_path)
    train_dataset = SentDataSet(train_df, tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=collate_func)

    val_df = load_data(args.dev_data_path)
    val_dataset = SentDataSet(val_df, tokenizer)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=args.dev_batch_size, collate_fn=collate_func)

    num_train_steps = int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # 模型
    model = Model()
    if torch.cuda.is_available():
        model.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    warmup_steps = 0.05 * num_train_steps
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            if torch.cuda.is_available():
                batch = (t.cuda() for t in batch)
            s1_input_ids, s2_input_ids, label_id = batch
            if torch.cuda.is_available():
                s1_input_ids, s2_input_ids, label_id = s1_input_ids.cuda(), s2_input_ids.cuda(), label_id.cuda()

            s1_vec = model(s1_input_ids, encoder_type='cls')
            s2_vec = model(s2_input_ids, encoder_type='cls')

            loss = calc_loss(s1_vec, s2_vec, label_id)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            print('Epoch:{}, Step:{}, Loss:{:10f}'.format(epoch, step, loss))

            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), max_norm=20, norm_type=2)   # 是否进行梯度裁剪

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # corrcoef, pearsonr = evaluate(model)
        # 一轮跑完 进行eval
        corrcoef, pearsonr = evaluate(model)
        ss = 'epoch:{}, spearmanr:{:10f}, pearsonr:{:10f}'.format(epoch, corrcoef, pearsonr)
        with open(args.output_dir + '/logs.txt', 'a+', encoding='utf8') as f:
            ss += '\n'
            f.write(ss)
        model.train()

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "epoch{}_ckpt.bin".format(epoch))
        torch.save(model_to_save.state_dict(), output_model_file)
