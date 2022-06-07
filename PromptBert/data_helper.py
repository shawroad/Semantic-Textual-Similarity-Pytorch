"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-06-07
"""
import torch
import pandas as pd
from config import set_args
from torch.utils.data import Dataset

args = set_args()


def load_data(path, tokenizer):
    # 定义的模板
    replace_token = "[X]"
    prompt_templates = ['"{}"，它的意思是[MASK]。'.format(replace_token), '"{}"，这句话的意思是[MASK]。'.format(replace_token)]

    sent_prompt1, sent_template1, sent_prompt2, sent_template2 = [], [], [], []
    with open(path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split('\t')[0]
            words_num = len(tokenizer.tokenize(line))

            if words_num > args.max_len - 15:
                # 因为模型最大字符为15个  所以在最大长度上要减去模板的长度 才是真正文本的长度
                line = line[:args.max_len - 15]
            line_num = len(tokenizer.tokenize(line))
            # 第一个模板
            prompt_line1 = prompt_templates[0].replace(replace_token, line)
            template_line1 = prompt_templates[0].replace(replace_token, replace_token * line_num)
            # 第二个模板
            prompt_line2 = prompt_templates[1].replace(replace_token, line)
            template_line2 = prompt_templates[1].replace(replace_token, replace_token * line_num)

            sent_prompt1.append(prompt_line1)
            sent_template1.append(template_line1)
            sent_prompt2.append(prompt_line2)
            sent_template2.append(template_line2)

    df = pd.DataFrame({'sent_prompt1': sent_prompt1, 'sent_template1': sent_template1,
                       'sent_prompt2': sent_prompt2, 'sent_template2': sent_template2})
    return df


class SentDataSet(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = data
        self.sent_prompt1 = self.data['sent_prompt1']
        self.sent_template1 = self.data['sent_template1']

        self.sent_prompt2 = self.data['sent_prompt2']
        self.sent_template2 = self.data['sent_template2']

        self.max_len = args.max_len

    def __len__(self):
        return len(self.sent_prompt1)

    def __getitem__(self, idx):
        sent_prompt1_input = self.tokenizer.encode_plus(
            text=self.sent_prompt1[idx],
            text_pair=None,
            add_special_tokens=True,
            return_token_type_ids=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        sent_prompt1_input_ids = sent_prompt1_input['input_ids']
        sent_prompt1_attention_mask = sent_prompt1_input['attention_mask']
        sent_prompt1_token_type_ids = sent_prompt1_input["token_type_ids"]

        sent_template1_input = self.tokenizer.encode_plus(
            text=self.sent_template1[idx],
            text_pair=None,
            add_special_tokens=True,
            return_token_type_ids=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        sent_template1_input_ids = sent_template1_input['input_ids']
        sent_template1_attention_mask = sent_template1_input['attention_mask']
        sent_template1_token_type_ids = sent_template1_input["token_type_ids"]

        sent_prompt2_input = self.tokenizer.encode_plus(
            text=self.sent_prompt2[idx],
            text_pair=None,
            add_special_tokens=True,
            return_token_type_ids=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        sent_prompt2_input_ids = sent_prompt2_input['input_ids']
        sent_prompt2_attention_mask = sent_prompt2_input['attention_mask']
        sent_prompt2_token_type_ids = sent_prompt2_input["token_type_ids"]

        sent_template2_input = self.tokenizer.encode_plus(
            text=self.sent_template2[idx],
            text_pair=None,
            add_special_tokens=True,
            return_token_type_ids=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        sent_template2_input_ids = sent_template2_input['input_ids']
        sent_template2_attention_mask = sent_template2_input['attention_mask']
        sent_template2_token_type_ids = sent_template2_input["token_type_ids"]

        return {'sent_prompt1_input_ids': sent_prompt1_input_ids,
                'sent_prompt1_attention_mask': sent_prompt1_attention_mask,
                'sent_prompt1_token_type_ids': sent_prompt1_token_type_ids,

                'sent_template1_input_ids': sent_template1_input_ids,
                'sent_template1_attention_mask': sent_template1_attention_mask,
                'sent_template1_token_type_ids': sent_template1_token_type_ids,

                'sent_prompt2_input_ids': sent_prompt2_input_ids,
                'sent_prompt2_attention_mask': sent_prompt2_attention_mask,
                'sent_prompt2_token_type_ids': sent_prompt2_token_type_ids,

                'sent_template2_input_ids': sent_template2_input_ids,
                'sent_template2_attention_mask': sent_template2_attention_mask,
                'sent_template2_token_type_ids': sent_template2_token_type_ids,
                }


def collate_func(batch_data):
    sent_prompt1_input_ids_list, sent_prompt1_attention_mask_list, sent_prompt1_token_type_ids_list = [], [], []
    sent_template1_input_ids_list, sent_template1_attention_mask_list, sent_template1_token_type_ids_list = [], [], []
    sent_prompt2_input_ids_list, sent_prompt2_attention_mask_list, sent_prompt2_token_type_ids_list = [], [], []
    sent_template2_input_ids_list, sent_template2_attention_mask_list, sent_template2_token_type_ids_list = [], [], []

    for item in batch_data:
        sent_prompt1_input_ids_list.append(item['sent_prompt1_input_ids'])
        sent_prompt1_attention_mask_list.append(item['sent_prompt1_attention_mask'])
        sent_prompt1_token_type_ids_list.append(item['sent_prompt1_token_type_ids'])

        sent_template1_input_ids_list.append(item['sent_template1_input_ids'])
        sent_template1_attention_mask_list.append(item['sent_template1_attention_mask'])
        sent_template1_token_type_ids_list.append(item['sent_template1_token_type_ids'])

        sent_prompt2_input_ids_list.append(item['sent_prompt2_input_ids'])
        sent_prompt2_attention_mask_list.append(item['sent_prompt2_attention_mask'])
        sent_prompt2_token_type_ids_list.append(item['sent_prompt2_token_type_ids'])

        sent_template2_input_ids_list.append(item['sent_template2_input_ids'])
        sent_template2_attention_mask_list.append(item['sent_template2_attention_mask'])
        sent_template2_token_type_ids_list.append(item['sent_template2_token_type_ids'])

    all_sent_prompt1_input_ids = torch.tensor(sent_prompt1_input_ids_list, dtype=torch.long)
    all_sent_prompt1_attention_mask = torch.tensor(sent_prompt1_attention_mask_list, dtype=torch.long)
    all_sent_prompt1_token_type_ids = torch.tensor(sent_prompt1_token_type_ids_list, dtype=torch.long)

    all_sent_template1_input_ids = torch.tensor(sent_template1_input_ids_list, dtype=torch.long)
    all_sent_template1_attention_mask = torch.tensor(sent_template1_attention_mask_list, dtype=torch.long)
    all_sent_template1_token_type_ids = torch.tensor(sent_template1_token_type_ids_list, dtype=torch.long)

    all_sent_prompt2_input_ids = torch.tensor(sent_prompt2_input_ids_list, dtype=torch.long)
    all_sent_prompt2_attention_mask = torch.tensor(sent_prompt2_attention_mask_list, dtype=torch.long)
    all_sent_prompt2_token_type_ids = torch.tensor(sent_prompt2_token_type_ids_list, dtype=torch.long)

    all_sent_template2_input_ids = torch.tensor(sent_template2_input_ids_list, dtype=torch.long)
    all_sent_template2_attention_mask = torch.tensor(sent_template2_attention_mask_list, dtype=torch.long)
    all_sent_template2_token_type_ids = torch.tensor(sent_template2_token_type_ids_list, dtype=torch.long)

    return (all_sent_prompt1_input_ids, all_sent_prompt1_attention_mask, all_sent_prompt1_token_type_ids,
            all_sent_template1_input_ids, all_sent_template1_attention_mask, all_sent_template1_token_type_ids,
            all_sent_prompt2_input_ids, all_sent_prompt2_attention_mask, all_sent_prompt2_token_type_ids,
            all_sent_template2_input_ids, all_sent_template2_attention_mask, all_sent_template2_token_type_ids)

