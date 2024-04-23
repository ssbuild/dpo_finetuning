
```text
    2024-04-22 简化
    2023-10-09 support accelerator trainer
    2023-10-07 support colossalai trainer
    2023-09-26 support transformers trainer
    2023-09-15 支持transformers系列模型 dpo训练  
```

## update information
   - [deep_training](https://github.com/ssbuild/deep_training)

## 支持训练方式

| 模型        | 32精度 | 混合精度 | 16精度 | lora int8 | lora int4 | ptv2   |
|-----------|-------|---------|-------|-----------|-----------|--------|
| llama     | √    | √    | √    | √         | √         | ×      |
| llama2    | √    | √    | √    | √         | √         | ×      |
| chatglm   | √    | √    | √    | √         | √         | ×      |
| chatglm2  | √    | √    | √    | √         | √         | ×      |
| baichuan  | √    | √    | √    | √         | √         | ×      |
| baichuan2 | √    | √    | √    | √         | √         | ×      |
| opt       | √    | √    | √    | √         | √         | ×      |
| moss      | √    | √    | √    | √         | √         | ×      |
| rwkv      | √    | √    | √    | √         | √         | ×      |
| tiger     | √    | √    | √    | √         | √         | ×      |
| qwen      | √    | √    | √    | √         | √         | ×      |
| internlm  | √    | √    | √    | √         | √         | ×      |
| gpt2      | √    | √    | √    | √         | √         | ×      |
| xverse    | √    | √    | √    | √         | √         | ×      |
| openbuddy | √    | √    | √    | √         | √         | ×      |


## 数据示例
```json
{
	"prompt": "\n\nHuman:你能描述如何清除笔记本计算机屏幕上的指纹和污垢的步骤吗?\n\nAssistant:是的,也许。为了清洁你的屏幕,你首先需要使用微纤维布或软溼布,轻轻地擦拭屏幕表面。接下来,你会想抓住一个软,无毛巾,微纤维清洁布,轻轻地擦回屏幕上去除指纹和污垢。\n\nHuman:我可以将伊索普罗素酒精喷射到布上,并以这种方式清洗它吗?\n\nAssistant:",
	"response": " 是的,你可以这样做,以帮助布从屏幕上收集更多的污垢。 请确保始终使用干净的软布,而不是一块粗糙的、粗糙的或结构化的材料,并确保它是无毛布的。",
	"chosen": " 是的,你可以这样做,以帮助布从屏幕上收集更多的污垢。 请确保始终使用干净的软布,而不是一块粗糙的、粗糙的或结构化的材料,并确保它是无毛布的。",
	"rejected": " 是的,你可以直接将它喷射到布上。"
}
```
    



## 生成训练record
   
- cd data && make_data_example.py 
- python data_utils.py
    
    注:
    num_process_worker 为多进程制作数据 ， 如果数据量较大 ， 适当调大至cpu数量
    dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)





## training
```text
    # 制作数据
    cd scripts
    bash train_full.sh -m dataset 
    or
    bash train_lora.sh -m dataset 
    or
    bash train_ptv2.sh -m dataset 
    
    注: num_process_worker 为多进程制作数据 ， 如果数据量较大 ， 适当调大至cpu数量
    dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)
    
    # 全参数训练 
        bash train_full.sh -m train
        
    # lora adalora ia3 
        bash train_lora.sh -m train
        
    # ptv2
        bash train_ptv2.sh -m train
```






## 训练参数
[训练参数](args.MD)

## 友情链接

- [pytorch-task-example](https://github.com/ssbuild/pytorch-task-example)
- [chatmoss_finetuning](https://github.com/ssbuild/chatmoss_finetuning)
- [chatglm_finetuning](https://github.com/ssbuild/chatglm_finetuning)
- [chatglm2_finetuning](https://github.com/ssbuild/chatglm2_finetuning)
- [t5_finetuning](https://github.com/ssbuild/t5_finetuning)
- [llm_finetuning](https://github.com/ssbuild/llm_finetuning)
- [llm_rlhf](https://github.com/ssbuild/llm_rlhf)
- [chatglm_rlhf](https://github.com/ssbuild/chatglm_rlhf)
- [t5_rlhf](https://github.com/ssbuild/t5_rlhf)
- [rwkv_finetuning](https://github.com/ssbuild/rwkv_finetuning)
- [baichuan_finetuning](https://github.com/ssbuild/baichuan_finetuning)
- [internlm_finetuning](https://github.com/ssbuild/internlm_finetuning)
- [qwen_finetuning](https://github.com/ssbuild/qwen_finetuning)
- [xverse_finetuning](https://github.com/ssbuild/xverse_finetuning)
- [auto_finetuning](https://github.com/ssbuild/auto_finetuning)
- [aigc_serving](https://github.com/ssbuild/aigc_serving)

## 
    纯粹而干净的代码


## 参考
- https://arxiv.org/pdf/2305.18290.pdf
- https://github.com/eric-mitchell/direct-preference-optimization