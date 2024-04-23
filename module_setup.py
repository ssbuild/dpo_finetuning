# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/9/19 16:22

import os
from deep_training.utils.hf import register_transformer_model, register_transformer_config, \
    register_transformer_tokenizer
from transformers import AutoModelForCausalLM
from deep_training.nlp.models.rellama.modeling_llama import LlamaForCausalLM
from config import config_args

__all__ = [
    "module_setup",
    "global_model_card"
]

global_model_card = (config_args["model_name_or_path"] or config_args["config_name"])
global_model_card = global_model_card.split('/')[-1].lower() if os.path.isdir(global_model_card) else \
    str(os.path.dirname(global_model_card)).split('/')[-1].lower()


def module_setup():
    register_transformer_model(LlamaForCausalLM, AutoModelForCausalLM)

    if "baichuan" in global_model_card:
        assert "7b" in global_model_card or "13b" in global_model_card
        if "baichuan2" in global_model_card:
            if "7b" in global_model_card:
                from deep_training.zoo.model_zoo.baichuan.baichuan2_7b.llm_model import (MyBaichuanForCausalLM as LM_MODEL, PetlArguments,  # noqa
                                                                                LoraConfig, PetlModel,
                                                                                PromptArguments,
                                                                                BaichuanConfig,
                                                                                BaichuanTokenizer)
            else:
                from deep_training.zoo.model_zoo.baichuan.baichuan2_13b.llm_model import (MyBaichuanForCausalLM as LM_MODEL, PetlArguments,  # noqa
                                                                                 LoraConfig, PetlModel,
                                                                                 PromptArguments,
                                                                                 BaichuanConfig,
                                                                                 BaichuanTokenizer)
        else:
            if "7b" in global_model_card:
                from deep_training.zoo.model_zoo.baichuan.baichuan_7b.llm_model import (MyBaichuanForCausalLM as LM_MODEL, PetlArguments,  # noqa
                                                                               LoraConfig, PetlModel,
                                                                               PromptArguments,
                                                                               BaichuanConfig,
                                                                               BaichuanTokenizer)
            else:
                from deep_training.zoo.model_zoo.baichuan.baichuan_13b.llm_model import (MyBaichuanForCausalLM as LM_MODEL, PetlArguments,  # noqa
                                                                                LoraConfig, PetlModel,
                                                                                PromptArguments,
                                                                                BaichuanConfig,
                                                                                BaichuanTokenizer)

        register_transformer_config(BaichuanConfig)
        register_transformer_model(LM_MODEL, AutoModelForCausalLM)
        register_transformer_tokenizer(BaichuanConfig,BaichuanTokenizer,BaichuanTokenizer)

    elif "xverse" in global_model_card:
        from deep_training.zoo.model_zoo.xverse.llm_model import (MyXverseForCausalLM as LM_MODEL,
                                                        PetlArguments,  # noqa
                                                        LoraConfig, PetlModel,
                                                        PromptArguments,
                                                        XverseConfig,)

        register_transformer_config(XverseConfig)
        register_transformer_model(LM_MODEL, AutoModelForCausalLM)

    elif "qwen" in global_model_card:
        from deep_training.zoo.model_zoo.qwen.llm_model import (MyQWenLMHeadModel as LM_MODEL,
                                                     PetlArguments,  # noqa
                                                     LoraConfig, PetlModel,
                                                     PromptArguments,
                                                     QWenConfig,QWenTokenizer )

        register_transformer_config(QWenConfig)
        register_transformer_model(LM_MODEL, AutoModelForCausalLM)
        register_transformer_tokenizer(QWenConfig, QWenTokenizer, QWenTokenizer)

    elif "internlm" in global_model_card:
        from deep_training.zoo.model_zoo.internlm.llm_model import (MyInternLMForCausalLM as LM_MODEL,
                                                     PetlArguments,  # noqa
                                                     LoraConfig, PetlModel,
                                                     PromptArguments,
                                                     InternLMConfig,InternLMTokenizer )

        register_transformer_config(InternLMConfig)
        register_transformer_model(LM_MODEL, AutoModelForCausalLM)
        register_transformer_tokenizer(InternLMConfig, InternLMTokenizer, InternLMTokenizer)

    elif "chatglm2" in global_model_card:
        from deep_training.zoo.model_zoo.chatglm2.llm_model import (MyChatGLMForConditionalGeneration as LM_MODEL,
                                                           PetlArguments,  # noqa
                                                           LoraConfig, PetlModel,
                                                           PromptArguments,
                                                           ChatGLMConfig, ChatGLMTokenizer)

        register_transformer_config(ChatGLMConfig)
        register_transformer_model(LM_MODEL, AutoModelForCausalLM)
        register_transformer_tokenizer(ChatGLMConfig, ChatGLMTokenizer, ChatGLMTokenizer)

    elif "chatglm" in global_model_card:
        from deep_training.zoo.model_zoo.chatglm.llm_model import (MyChatGLMForConditionalGeneration as LM_MODEL,
                                                           PetlArguments,  # noqa
                                                           LoraConfig, PetlModel,
                                                           PromptArguments,
                                                           ChatGLMConfig, ChatGLMTokenizer)

        register_transformer_config(ChatGLMConfig)
        register_transformer_model(LM_MODEL, AutoModelForCausalLM)
        register_transformer_tokenizer(ChatGLMConfig, ChatGLMTokenizer, ChatGLMTokenizer)

    elif "moss" in global_model_card:
        from deep_training.zoo.model_zoo.moss.llm_model import (MyMossForCausalLM as LM_MODEL,
                                                      PetlArguments,  # noqa
                                                      LoraConfig, PetlModel,
                                                      PromptArguments,
                                                      MossConfig, MossTokenizer)

        register_transformer_config(MossConfig)
        register_transformer_model(LM_MODEL, AutoModelForCausalLM)
        register_transformer_tokenizer(MossConfig, MossTokenizer, MossTokenizer)


    elif "skywork" in global_model_card:
        from deep_training.zoo.model_zoo.skywork.llm_model import (MySkyworkForCausalLM as LM_MODEL,
                                                            PetlArguments,  # noqa
                                                            LoraConfig, PetlModel,
                                                            PromptArguments,
                                                            SkyworkConfig, SkyworkTokenizer)

        register_transformer_config(SkyworkConfig)
        register_transformer_model(LM_MODEL, AutoModelForCausalLM)
        register_transformer_tokenizer(SkyworkConfig,SkyworkTokenizer,SkyworkTokenizer)

    elif "bluelm" in global_model_card:
        from deep_training.zoo.model_zoo.bluelm.llm_model import (MyBlueLMForCausalLM as LM_MODEL,
                                                          PetlArguments,  # noqa
                                                          LoraConfig, PetlModel,
                                                          PromptArguments,
                                                          BlueLMConfig, BlueLMTokenizer)

        register_transformer_config(BlueLMConfig)
        register_transformer_model(LM_MODEL, AutoModelForCausalLM)
        register_transformer_tokenizer(BlueLMConfig, BlueLMTokenizer, BlueLMTokenizer)

    elif "yi" in global_model_card:
        from deep_training.zoo.model_zoo.yi.llm_model import (MyYiForCausalLM as LM_MODEL,
                                                         PetlArguments,  # noqa
                                                         LoraConfig, PetlModel,
                                                         PromptArguments,
                                                         YiConfig, YiTokenizer)

        register_transformer_config(YiConfig)
        register_transformer_model(LM_MODEL, AutoModelForCausalLM)
        register_transformer_tokenizer(YiConfig, YiTokenizer, YiTokenizer)
    elif "rwkv" in global_model_card:
        from deep_training.zoo.model_zoo.rwkv4.llm_model import (MyRwkvForCausalLM as LM_MODEL,
                                                       PetlArguments,  # noqa
                                                       LoraConfig, PetlModel,
                                                       PromptArguments,
                                                       RwkvConfig)
        register_transformer_config(RwkvConfig)
        register_transformer_model(LM_MODEL, AutoModelForCausalLM)
        if 'world' in global_model_card:
            from deep_training.zoo.model_zoo.rwkv4.rwkv4_tokenizer import RWKVTokenizer
            register_transformer_tokenizer(RwkvConfig, RWKVTokenizer, RWKVTokenizer)

    # 按需加入其他自定义模型