"""
@file   : esimcse.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-10-25
"""
import torch
import random
from torch import nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


class Model(nn.Module):
    def __init__(self, q_size=128, dup_rate=0.32, temperature=0.05, gamma=0.99):
        super(Model, self).__init__()
        self.config = BertConfig.from_pretrained('./roberta_pretrain/config.json')
        self.bert = BertModel.from_pretrained('./roberta_pretrain/pytorch_model.bin', config=self.config)
     
        # 下面这个是为了获取上一个batch中的样本编码向量
        self.moco_config = BertConfig.from_pretrained('./roberta_pretrain/config.json')
        self.moco_config.hidden_dropout_prob = 0.0  # 不用dropout
        self.moco_config.attention_probs_dropout_prob = 0.0    # 不用dropout
        self.moco_bert = BertModel.from_pretrained('./roberta_pretrain/pytorch_model.bin', config=self.moco_config)
    
        self.gamma = gamma  
        self.q = []   # 积攒负样本的队列
        self.q_size = q_size   # 队列长度
        self.dup_rate = dup_rate   # 数据增广的比例
        self.temperature = temperature   # 损失 热度
        self.loss_fct = nn.CrossEntropyLoss()
    
    def cal_cos_sim(self, embedding1, embedding2):
        embedding1_norm = F.normalize(embedding1, p=2, dim=1)
        embedding2_norm = F.normalize(embedding2, p=2, dim=1)
        return torch.mm(embedding1_norm, embedding2_norm.transpose(0, 1))  # (batch_size, batch_size)

    def word_repetition(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.size()
        input_ids, attention_mask = input_ids.cpu().tolist(), attention_mask.cpu().tolist()
        repetitied_input_ids, repetitied_attention_mask = [], []
        rep_seq_len = seq_len
        for batch_id in range(batch_size):
            # 一个一个序列进行处理
            sample_mask = attention_mask[batch_id]
            actual_len = sum(sample_mask)   # 计算当前序列的真实长度
            
            cur_input_ids = input_ids[batch_id]
            # 随机选取dup_len个token
            dup_len = random.randint(a=0, b=max(2, int(self.dup_rate * actual_len)))   # dup_rate越大  可能重复的token越多 否则越少
            dup_word_index = random.sample(list(range(1, actual_len)), k=dup_len)    #  采样出dup_len个token  然后下面进行重复
            r_input_id, r_attention_mask = [], []
            for index, word_id in enumerate(cur_input_ids):
                if index in dup_word_index:
                    r_input_id.append(word_id)
                    r_attention_mask.append(sample_mask[index])
                r_input_id.append(word_id)
                r_attention_mask.append(sample_mask[index])
            
            after_dup_len = len(r_input_id)
            repetitied_input_ids.append(r_input_id)
            repetitied_attention_mask.append(r_attention_mask)

            assert after_dup_len == dup_len + seq_len
            if after_dup_len > rep_seq_len:
                rep_seq_len = after_dup_len

        for i in range(batch_size):
            after_dup_len = len(repetitied_input_ids[i])
            pad_len = rep_seq_len - after_dup_len
            repetitied_input_ids[i] += [0] * pad_len
            repetitied_attention_mask[i] += [0] * pad_len

        repetitied_input_ids = torch.tensor(repetitied_input_ids, dtype=torch.long).cuda()
        repetitied_attention_mask = torch.tensor(repetitied_attention_mask, dtype=torch.long).cuda()
        return repetitied_input_ids, repetitied_attention_mask

    def forward(self, input_ids1, attention_mask1):
        # 这里直接取CLS向量 也可用其他的方式
        s1_embedding = self.bert(input_ids1, attention_mask1, output_hidden_states=True).last_hidden_state[:, 0]

        # 给当前输入的样本拷贝一个正样本
        input_ids2, attention_mask2 = torch.clone(input_ids1), torch.clone(attention_mask1)
        input_ids2, attention_mask2 = self.word_repetition(input_ids2, attention_mask2)   # 数据增广 重复某些字
        s2_embedding = self.bert(input_ids2, attention_mask2, output_hidden_states=True).last_hidden_state[:, 0]
        
        # 计算cos
        cos_sim = self.cal_cos_sim(s1_embedding, s2_embedding) / self.temperature   # (batch_size, batch_size)
        
        batch_size = cos_sim.size(0)
        assert cos_sim.size() == (batch_size, batch_size)
        
        negative_samples = None
        if len(self.q) > 0:
            # 从队列中取出负样本
            negative_samples = torch.cat(self.q[:self.q_size], dim=0)   # (q_size, 768)
        
        if len(self.q) + batch_size >= self.q_size:
            # 这个批次的样本准备加入到负样本队列  测试一下  加入进去 是否超过最大队列长度 如果超过 将队头多余的出队
            del self.q[:batch_size]
        
        # 将当前batch作为负样本 加入到负样本队列
        with torch.no_grad():
            self.q.append(self.moco_bert(input_ids1, attention_mask1, output_hidden_states=True).last_hidden_state[:, 0])

        labels = torch.arange(batch_size).cuda()
        if negative_samples is not None:
            batch_size += negative_samples.size(0)   # batch_size + 负样本的个数
            cos_sim_with_neg = self.cal_cos_sim(s1_embedding, negative_samples) / self.temperature  # 当前batch和之前负样本的cos (N, M)
            cos_sim = torch.cat([cos_sim, cos_sim_with_neg], dim=1)  # (N, N+M)
            
        for encoder_param, moco_encoder_param in zip(self.bert.parameters(), self.moco_bert.parameters()):
            moco_encoder_param.data = self.gamma * moco_encoder_param.data + (1. - self.gamma) * encoder_param.data
       
        loss = self.loss_fct(cos_sim, labels)
        return loss

    def encode(self, input_ids, attention_mask):
        s1_embedding = self.bert(input_ids, attention_mask, output_hidden_states=True).last_hidden_state[:, 0]
        return s1_embedding
  
