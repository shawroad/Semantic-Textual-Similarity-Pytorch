"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-02-23
"""
import torch
from torch import nn
from config import set_args
from transformers.models.bert import BertModel, BertConfig

args = set_args()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.config = BertConfig.from_pretrained(args.bert_pretrain_path)
        self.bert = BertModel.from_pretrained(args.bert_pretrain_path)
        self.clssify = nn.Linear(self.config.hidden_size * 3, 2)
    
    def get_embedding(self, output, encoder_type):
        if encoder_type == 'fist-last-avg':
            # 第一层和最后一层的隐层取出  然后经过平均池化
            first = output.hidden_states[1]   # hidden_states列表有13个hidden_state，第一个其实是embeddings，第二个元素才是第一层的hidden_state
            last = output.hidden_states[-1]
            seq_length = first.size(1)   # 序列长度

            first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            final_encoding = torch.avg_pool1d(torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1, 2), kernel_size=2).squeeze(-1)
            return final_encoding

        if encoder_type == 'last-avg':
            sequence_output = output.last_hidden_state  # (batch_size, max_len, hidden_size)
            seq_length = sequence_output.size(1)
            final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            return final_encoding

        if encoder_type == "cls":
            sequence_output = output.last_hidden_state
            cls = sequence_output[:, 0]  # [b,d]
            return cls

        if encoder_type == "pooler":
            pooler_output = output.pooler_output  # [b,d]
            return pooler_output
        

    def forward(self, s1_input_ids, s2_input_ids, encoder_type='cls'):
        s1_attention_mask = torch.ne(s1_input_ids, 0)
        s2_attention_mask = torch.ne(s2_input_ids, 0)
        
        s1_output = self.bert(s1_input_ids, s1_attention_mask, output_hidden_states=True)
        s1_embedding = self.get_embedding(s1_output, encoder_type)
        
        s2_output = self.bert(s2_input_ids, s2_attention_mask, output_hidden_states=True)
        s2_embedding = self.get_embedding(s2_output, encoder_type)
        
        diff = torch.abs(s1_embedding - s2_embedding)
        concat_vector = torch.cat([s1_embedding, s2_embedding, diff], dim=-1)
        
        logits = self.clssify(concat_vector)
        return logits
        
    def encode(self, input_ids, encoder_type='cls'):
        attention_mask = torch.ne(input_ids, 0)
        output = self.bert(input_ids, attention_mask, output_hidden_states=True)
        embedding = self.get_embedding(output, encoder_type)
        return embedding
        


