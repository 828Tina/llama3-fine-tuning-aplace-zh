from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel
import torch
from swanlab.integration.huggingface import SwanLabCallback
from trl import SFTTrainer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.device_count()

swanlab_callback = SwanLabCallback(
    project="llama3-fintune",
    experiment_name="llama3-8b",
    description="使用llama3-8b在alpaca_data_zh_51k.json数据集上微调",
)
train_params = TrainingArguments(
    optim="paged_adamw_32bit",
    learning_rate=3e-4,
    weight_decay=0.01,
    lr_scheduler_type='cosine',
    warmup_ratio=0.03,
    gradient_accumulation_steps=4,
    bf16=True,
    gradient_checkpointing=True,
    label_smoothing_factor=0.1,
    neftune_noise_alpha=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    max_grad_norm=2,
    group_by_length=True,
    num_train_epochs=3,
    output_dir='./output',
    save_steps=500,
    logging_steps=10
)

max_seq_length = 1024
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 准备微调数据集
EOS_TOKEN = tokenizer.eos_token  # 必须添加 EOS_TOKEN

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

Instruction:
{}

Input:
{}

Response:
{}"""


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # 必须添加EOS_TOKEN，否则无限生成
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts, }


dataset = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train")
tune_data = dataset.map(formatting_prompts_func, batched=True, )

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tune_data,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=train_params,
    callbacks=[swanlab_callback]
)

trainer.train()
model.save_pretrained("lora_model")  # Local saving
model.save_pretrained_merged("outputs", tokenizer, save_method="merged_16bit", )  # 合并模型，保存为16位hf
model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
