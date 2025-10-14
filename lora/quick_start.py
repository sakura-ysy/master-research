from unsloth import FastLanguageModel
import torch
from torch.utils.data import Dataset
from transformers import (
    TrainingArguments,
)
from trl import SFTTrainer
from datasets import load_dataset
import torch

# 一些基本配置
model_name = "/data/llm/Qwen3-8B"
data_path = "/home/ysy/code/LoRA/data" # 指令微调数据集, 52k
sample_size = 1000 # 只用 1000 条数据训练
output_dir = "/home/ysy/lora_model/qwen3-8B-lora-custom"


# 加载预训练模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/data/llm/Qwen3-8B",  # 使用Qwen3-8B模型
    max_seq_length = 4096,
    trust_remote_code = True,
)

# 添加LoRA适配器
model = FastLanguageModel.get_peft_model(
    model,
    r = 8,  # LoRA秩，建议使用8、16、32、64、128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],  # 需要应用LoRA的模块
    lora_alpha = 16,  # LoRA缩放因子
    lora_dropout = 0,  # LoRA dropout率，0为优化设置
    bias = "none",    # 偏置项设置，none为优化设置
    use_gradient_checkpointing = "unsloth",  # 使用unsloth的梯度检查点，可减少30%显存使用
    random_state = 3407,  # 随机种子
    use_rslora = False,  # 是否使用rank stabilized LoRA
    loftq_config = None,  # LoftQ配置
)

model.print_trainable_parameters()

# Alpaca 格式的 prompt 模板
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

# 结束标记
EOS_TOKEN = tokenizer.eos_token


def process_data(data: dict, tokenizer, max_seq_length):
    input_ids, attention_mask, labels = [], [], []
    # 指令微调的数据
    instruction_text = data['instruction']
    human_text = data["input"]
    output_text = data["output"]
 
    input_text = alpaca_prompt.format(instruction_text, human_text)
 
    input_tokenizer = tokenizer(
        input_text,
        add_special_tokens=False,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    output_tokenizer = tokenizer(
        output_text,
        add_special_tokens=False,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
 
    input_ids += (
            input_tokenizer["input_ids"] + output_tokenizer["input_ids"] + [tokenizer.eos_token_id]
    )
    attention_mask += input_tokenizer["attention_mask"] + output_tokenizer["attention_mask"] + [1]
    labels += ([-100] * len(input_tokenizer["input_ids"]) + output_tokenizer["input_ids"] + [tokenizer.eos_token_id]
               )
 
    if len(input_ids) > max_seq_length:  # 做一个截断
        input_ids = input_ids[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
        labels = labels[:max_seq_length]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 加载数据集
train_dataset = load_dataset(path=data_path, split="train")
# train_dataset = train_dataset.map(formatting_prompts_func, batched = True)
train_dataset = train_dataset.shuffle(seed=42).select(range(1000))
train_dataset = train_dataset.map(process_data,
                             fn_kwargs={"tokenizer": tokenizer, "max_seq_length": 2048},
                             remove_columns=train_dataset.column_names)

# ======= 训练参数 =======
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    warmup_ratio=0.03,
    gradient_checkpointing=True,
    optim="adamw_torch",
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    report_to="none",
)

# ======= Trainer =======
# 会自动去找 dataset 中的 'text' 字段进行训练
trainer = SFTTrainer(
  model=model,
  args=training_args,
  train_dataset=train_dataset,
  processing_class=tokenizer, # In the 0.12.0 release it is explained that the tokenzier argument is now called the processing_class parameter.
)

trainer.train()

# ======= 保存 =======
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"LoRA finetune success, save model parameters to {output_dir}")
