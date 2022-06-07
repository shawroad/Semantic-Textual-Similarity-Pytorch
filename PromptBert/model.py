"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-06-07
"""
from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch
import torch.nn as nn
from transformers.models.bert import BertModel, BertConfig


class PromptBERT(nn.Module):
    def __init__(self, mask_id):
        super(PromptBERT, self).__init__()
        self.config = BertConfig.from_pretrained('./roberta_pretrain')
        # 可以根据要求制定dropout_rate
        # self.config.attention_probs_dropout_prob = 0.1
        # self.config.hidden_dropout_prob = 0.1
        self.bert = BertModel.from_pretrained('./roberta_pretrain', config=self.config)
        self.mask_id = mask_id

    def forward(self, prompt_input_ids, prompt_attention_mask, prompt_token_type_ids,
                template_input_ids, template_attention_mask, template_token_type_ids):
        prompt_embedding = self.calc_mask_embedding(prompt_input_ids, prompt_attention_mask, prompt_token_type_ids)
        # print(prompt_embedding.size())   # torch.Size([4, 768])

        template_embedding = self.calc_mask_embedding(template_input_ids, template_attention_mask, template_token_type_ids)
        sent_embedding = prompt_embedding - template_embedding
        return sent_embedding

    def calc_mask_embedding(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        token_embeddings = output[0]    # torch.Size([4, 48, 768])
        mask_index = (input_ids == self.mask_id).long()   # 找出mask的那个位置 取出其向量
        # print(mask_index.size())   # batch_size, max_len
        mask_embedding = self.get_mask_embedding(token_embeddings, mask_index)
        return mask_embedding

    def get_mask_embedding(self, token_embeddings, mask_index):
        input_mask_expanded = mask_index.unsqueeze(-1).expand(token_embeddings.size()).float()
        # print(input_mask_expanded.size())    # torch.Size([4, 48, 768])
        mask_embedding = torch.sum(token_embeddings * input_mask_expanded, 1)
        return mask_embedding