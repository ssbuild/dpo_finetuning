# @Time    : 2023/4/19 23:02
# @Author  : tk
# @FileName: data_utils
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from collections import OrderedDict

from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments, TrainingArgumentsHF, \
    TrainingArgumentsCL, TrainingArgumentsAC
from deep_training.nlp.models.petl import PetlArguments
from transformers import HfArgumentParser
from data_factory.data_helper_loader import (NN_DataHelper_Base,
                                             NN_DataHelper_baichuan,
                                             NN_DataHelper_chatglm,
                                             NN_DataHelper_chatglm2,
                                             NN_DataHelper_bloom,
                                             NN_DataHelper_internlm,
                                             NN_DataHelper_gpt2,
                                             NN_DataHelper_llama,
                                             NN_DataHelper_moss,
                                             NN_DataHelper_moss_plugin,
                                             NN_DataHelper_opt,
                                             NN_DataHelper_qwen,
                                             NN_DataHelper_tiger,
                                             NN_DataHelper_xverse,
                                             NN_DataHelper_rwkv,
                                             NN_DataHelper_openbuddy)

from config import *
from module_setup import global_model_card


def _find_data_helper():
    data_helper_mapper = OrderedDict({
        "baichuan": NN_DataHelper_baichuan,
        "chatglm2": NN_DataHelper_chatglm2,
        "chatglm": NN_DataHelper_chatglm,
        "xverse": NN_DataHelper_xverse,
        "qwen": NN_DataHelper_qwen,
        "gpt2": NN_DataHelper_gpt2,
        "llama": NN_DataHelper_llama,
        "internlm": NN_DataHelper_internlm,
        "opt": NN_DataHelper_opt,
        "bloom": NN_DataHelper_bloom,
        "tiger": NN_DataHelper_tiger,
        "moss": NN_DataHelper_moss,
        "moss_plugin": NN_DataHelper_moss_plugin,
        "rwkv": NN_DataHelper_rwkv,
        "openbuddy": NN_DataHelper_openbuddy,
        "default": NN_DataHelper_Base,
    })
    NN_DataHelper = None
    for k in data_helper_mapper:
        if k in global_model_card:
            if k == "moss" and "plugin" in global_model_card:
                k = "moss_plugin"
            NN_DataHelper = data_helper_mapper[ k ]
            break
    return NN_DataHelper


NN_DataHelper = _find_data_helper()
if NN_DataHelper is None:
    NN_DataHelper = NN_DataHelper_Base
    raise ValueError(f"{global_model_card} for data_helper is not implemented ")

if __name__ == '__main__':
    if global_args[ "trainer_backend" ] == "hf":
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsHF, DataArguments, PetlArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args = parser.parse_dict(config_args,
                                                                                         allow_extra_keys=True, )
    elif global_args[ "trainer_backend" ] == "pl":
        parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, PetlArguments))
        model_args, training_args, data_args, _ = parser.parse_dict(config_args)
    elif global_args["trainer_backend"] == "ac":
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsCL, DataArguments, PetlArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args = parser.parse_dict(config_args, allow_extra_keys=True, )
    else:
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsAC, DataArguments, PetlArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args = parser.parse_dict(config_args, allow_extra_keys=True, )

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(config_kwargs={"torch_dtype": torch.float16})

    # 缓存数据集
    print(f'to make dataset is overwrite_cache {data_args.overwrite_cache}')
    dataHelper.make_dataset_all()

    print('make dataset complete!')
    print('check data !')
    dataset = dataHelper.load_sequential_sampler(dataHelper.load_dataset_files()["train_files"],
                                                 with_load_memory=data_args.data_backend == 'record',
                                                 batch_size=1,
                                                 collate_fn=dataHelper.collate_fn)

    print('total', len(dataset))
    for i, d in enumerate(dataset):
        print(d)
        if i > 3:
            break