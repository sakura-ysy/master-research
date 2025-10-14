from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizerFast
from peft import PeftModel  # 0.5.0
import torch

# Load Models
base_model = "/data/llm/Meta-Llama-3-8B"
peft_model = "/data/llm/fingpt-mt_llama3-8b_lora"
tokenizer = LlamaTokenizerFast.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map = "cuda:0")
model = PeftModel.from_pretrained(model, peft_model)
model = model.merge_and_unload()
save_dir = "/data/llm/fingpt-mt_llama3-8b_merged"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print("Model merged and saved to", save_dir)