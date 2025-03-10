# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/9/25 12:29
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import copy
import logging
import math
import datasets
import torch
import transformers
from deep_training.trainer.ac.trainer import TrainerAC
from transformers import (
    HfArgumentParser,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from data_utils import NN_DataHelper, config_args, get_deepspeed_config, global_args
from deep_training.zoo.model_zoo.auto.dpo_model import MyTransformerDPO,PetlArguments, LoraConfig
from deep_training.data_helper import ModelArguments, DataArguments,TrainingArgumentsAC

from module_setup import global_model_card

assert global_args["trainer_backend"] == "ac"

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.33.2")

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def main():
    training_args: TrainingArgumentsAC
    parser = HfArgumentParser((ModelArguments, TrainingArgumentsAC, DataArguments, PetlArguments),
                              conflict_handler='resolve')
    model_args, training_args, data_args, lora_args = parser.parse_dict(config_args,allow_extra_keys=True,)
    lora_args = lora_args.config

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    config_kwargs = {"torch_dtype": torch.float16}
    if global_args['config_merge']:
        config_kwargs.update(global_args['config_merge'])

    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(config_kwargs=config_kwargs)

    with training_args.main_process_first(desc="make_dataset_all"):
        dataHelper.make_dataset_all()

    is_bf16_supported = torch.cuda.is_bf16_supported()
    precision = global_args[ "precision" ]
    if precision == "auto":
        # 精度 根据实际情况做调整
        if is_bf16_supported:
            precision = 'bf16'
        else:
            precision = '16'

        if global_args["quantization_config"] is not None and global_args["quantization_config"].load_in_8bit:
            precision = "32"


    if str(precision) == '16':
        training_args.fp16 = True
    elif str(precision) == 'bf16':
        training_args.bf16 = True
    else:
        training_args.fp16 = False
        training_args.bf16 = False

    if 'rwkv' in global_model_card:
        if precision.startswith('16'):
            RWKV_FLOAT_MODE = '16'
        elif precision.startswith('bf16'):
            RWKV_FLOAT_MODE = 'bf16'
        else:
            RWKV_FLOAT_MODE = '32'
        from deep_training.nlp.models.rwkv4.modeling_rwkv import set_model_profile
        # 加载cuda_core
        set_model_profile(RWKV_T_MAX=config.ctx_len, RWKV_FLOAT_MODE=RWKV_FLOAT_MODE)

    deepspeed_config = get_deepspeed_config(precision)
    if deepspeed_config:
        training_args.deepspeed = deepspeed_config

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    world_size,local_rank,process_index = training_args.world_size,training_args.local_rank,training_args.process_index

    dpo_args = dict(beta=0.1, ref_free=False)
    transformer_args = dict(config=config, model_args=model_args, training_args=training_args, lora_args=lora_args,
                            quantization_config=global_args["quantization_config"],
                            device_map={"": local_rank} if world_size > 1 else "auto",
                            torch_dtype=torch.float16,
                            new_num_tokens=len(tokenizer),  # 可能扩充词
                            **dpo_args
                            )

    if transformer_args["quantization_config"] is None:
        transformer_args.pop("device_map")

    pl_model = MyTransformerDPO(**transformer_args)

    pl_ref_model = copy.deepcopy(pl_model)
    pl_ref_model = pl_ref_model.eval().half()
    pl_ref_model.requires_grad_(False)

    pl_model.backbone.set_ref_model(pl_ref_model)

    config.save_pretrained(training_args.output_dir)

    # 加载sft权重
    # pl_model.load_sft_weight('./best_ckpt/best.pt',is_trainable=True)

    pl_model = pl_model.float()

    train_datasets = None
    if training_args.do_train:
        train_datasets = dataHelper.load_distributed_random_sampler(
            dataHelper.load_dataset_files()["train_files"],
            with_load_memory=data_args.data_backend == 'record',
            collate_fn=dataHelper.collate_fn,
            batch_size=training_args.per_device_train_batch_size,
            drop_last=training_args.dataloader_drop_last,  # 多卡建议扔掉
            num_processes=world_size, process_index=process_index,
            num_workers = training_args.dataloader_num_workers,
            pin_memory = training_args.dataloader_pin_memory,
        )



    # Initialize our Trainer
    trainer = TrainerAC(
        model=pl_model,
        args=training_args,
        train_dataset=train_datasets,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)




def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
