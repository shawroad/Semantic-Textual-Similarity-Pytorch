import torch
import random
from torch import nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.config = BertConfig.from_pretrained('./roberta_pretrain/config.json')
        self.bert = BertModel.from_pretrained('./roberta_pretrain/pytorch_model.bin', config=self.config)

        self.loss_fct = nn.CrossEntropyLoss()
        self.temperature = 0.05
        self.mean, self.std = 0, 1
        self.reg_size = 32   # 随机构造的负样本个数 
    
    def cal_cos_sim(self, embedding1, embedding2):
        embedding1_norm = F.normalize(embedding1, p=2, dim=1)
        embedding2_norm = F.normalize(embedding2, p=2, dim=1)
        return torch.mm(embedding1_norm, embedding2_norm.transpose(0, 1))  # (batch_size, batch_size)

    def forward(self, input_ids1, attention_mask1):
        s1_embedding = self.bert(input_ids1, attention_mask1, output_hidden_states=True).last_hidden_state[:, 0]
        # 拷贝一份数据 作为正样本
        input_ids2, attention_mask2 = torch.clone(input_ids1), torch.clone(attention_mask1)
        s2_embedding = self.bert(input_ids2, attention_mask2, output_hidden_states=True).last_hidden_state[:, 0]
        
        cos_sim = self.cal_cos_sim(s1_embedding, s2_embedding) / self.temperature   # (batch_size, batch_size)

        mean, std = 0, 1
        reg_size = 32   # 随便定义  看给它增加多少负样本      
        hidden_size = 768 
        reg_random = torch.normal(self.mean, self.std, size=(reg_size, s1_embedding.size(1))).cuda()
        # print(s1_embedding.size())   # torch.Size([32, 768]) 
        # print(reg_random.size())  # torch.Size([32, 768])
       
        reg_cos_sim = self.cal_cos_sim(s1_embedding, reg_random) / self.temperature
        # print(reg_cos_sim.size())  # torch.Size([32, 32]) 
        
        cos_sim = torch.cat((cos_sim, reg_cos_sim), dim=1)
        # print(cos_sim.size())  # torch.Size([32, 64])
        batch_size = cos_sim.size(0)
        labels = torch.arange(batch_size).cuda()
        loss = self.loss_fct(cos_sim, labels)
        return loss

    def encode(self, input_ids, attention_mask):
        s1_embedding = self.bert(input_ids, attention_mask, output_hidden_states=True).last_hidden_state[:, 0]
        return s1_embedding

