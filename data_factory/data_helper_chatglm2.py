# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/9/22 9:06
from typing import Any
import numpy as np
import torch
from deep_training.zoo.model_zoo.chatglm2.llm_model import build_masks_and_position_ids_glm
from torch.nn import functional as F
from transformers import PreTrainedTokenizer
from data_factory.data_helper_base import NN_DataHelper_Base, data_conf
from data_factory.data_processer import TokenIdsMakerForGLM2


class NN_DataHelper_chatglm2(NN_DataHelper_Base):
    def on_data_process(self, data: Any, mode: str):
        self.index += 1

        tokenizer: PreTrainedTokenizer
        config = self.config
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer

        pair_data = data

        if "sptoken" not in data_conf:
            data_conf["sptoken"] = tokenizer.encode("", add_special_tokens=True)

        d = TokenIdsMakerForGLM2.process(pair_data, tokenizer, max_seq_length, **data_conf)
        if self.index < 3:
            print(d)
        return d

    def collate_fn(self, batch):
        o = {k: [] for k in batch[0].keys()}
        for i, b in enumerate(batch):
            for k in b:
                o[k].append(torch.tensor(b[k]))


        seqlen = np.max([len(_) for _ in o['input_ids']])
        if 'input_ids2' in o:
            seqlen = np.max([seqlen] + [len(_) for _ in o['input_ids2']])

        tokenizer: PreTrainedTokenizer = self.tokenizer
        for k, v in o.items():
            pad_val = tokenizer.pad_token_id if 'label' not in k else -100
            o[k] = torch.stack(
                [F.pad(_, (0, seqlen - len(_)), mode='constant', value=pad_val) for _ in v])
        max_len = seqlen

        input_ids = o['input_ids']
        attention_mask, position_ids = build_masks_and_position_ids_glm(input_ids,  max_len)
        o['attention_mask'] = attention_mask.bool()
        o['position_ids'] = position_ids.long()
        o['labels'] = o["labels"].long()

        input_ids2 = o['input_ids2']
        attention_mask, position_ids = build_masks_and_position_ids_glm(input_ids2,  max_len)
        o['attention_mask2'] = attention_mask.bool()
        o['position_ids2'] = position_ids.long()
        o['labels2'] = o["labels2"].long()
        return o