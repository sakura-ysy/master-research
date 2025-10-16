import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizerFast, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import hash_prompt_sha256

def check_world_size(v: str) -> int:
    value = int(v)
    if value < 1:
        raise argparse.ArgumentTypeError("world_size must >= 1")
    if value != 1 and value % 2 != 0:
        raise argparse.ArgumentTypeError("world_size must be 1 or a multiple of 2")
    return value

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str.lower, default=None)
    parser.add_argument('--dataset', type=str.lower, nargs='+', default=["2wikimqa"], help="Specify one or more datasets to evaluate")
    parser.add_argument('--world_size', type=check_world_size, default=torch.cuda.device_count(), help="Number of GPUs to use, must be 1 or a multiple of 2")
    parser.add_argument('--long_bench_ds_path', type=str, default="THUDM/LongBench", help="Path to the LongBench dataset if you have it locally")
    parser.add_argument('--dump_qkv', action='store_true', help="Whether to dump kv cache")
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, kv_out_dir):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # 注释掉 apply_chat_template，为了使不同的 model 输入一样，好去做 kvcache diff
        # prompt = tokenizer.apply_chat_template([
        #     {"role": "user", "content": prompt},
        # ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        # if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
        #     prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)

        hash_key = hash_prompt_sha256(prompt, bits=256, encoding="hex")
        kv_meta_path = os.path.join(kv_out_dir, f"meta.json")
        with open(kv_meta_path, "a", encoding="utf-8") as f:
            json.dump({"_id": hash_key, "input": prompt}, f, ensure_ascii=False)
            f.write('\n')

        context_length = input.input_ids.shape[-1]
        if dataset == "samsum" or dataset == "2wikimqa": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                do_sample=False,
                temperature=None,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                do_sample=False,
                temperature=None,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        
        llm_layers = model.model.layers if hasattr(model, 'model') else model.layers
        os.makedirs(kv_out_dir,exist_ok=True)
        for j in range(len(llm_layers)):
            if hasattr(llm_layers[j], 'self_attn') and hasattr(llm_layers[j].self_attn, 'hack_qkv'):
                hack_qkv = llm_layers[j].self_attn.hack_qkv
                key_out_dir = os.path.join(kv_out_dir, f"{hash_key}")
                os.makedirs(key_out_dir, exist_ok=True)
                os.makedirs(os.path.join(key_out_dir, f"layer_{j}"), exist_ok=True)  # create layer_j
                torch.save(hack_qkv[0].cpu(), os.path.join(key_out_dir, f"layer_{j}", f"key.pt"))
                torch.save(hack_qkv[1].cpu(), os.path.join(key_out_dir, f"layer_{j}", f"value.pt"))
                torch.save(hack_qkv[2].cpu(), os.path.join(key_out_dir, f"layer_{j}", f"query.pt"))
                del llm_layers[j].self_attn.hack_qkv  # clear hack to save memory

    # dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "llama" in model_name:
        tokenizer = LlamaTokenizerFast.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import load_model
        model, _ = load_model(
            path,
            device='cpu',
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    elif "qwen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    datasets = args.dataset
    long_bench_ds_path = args.long_bench_ds_path
    # args.dataset can be a list of dataset names
    # define your model
    max_length = model2maxlen[model_name]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("kvcache"):
        os.makedirs("kvcache")

    for dataset in datasets:
        data = load_dataset(long_bench_ds_path, f"{dataset}", split='test')
        if not os.path.exists(f"kvcache/{model_name}"):
            os.makedirs(f"kvcache/{model_name}")
        kv_out_dir = f"kvcache/{model_name}/{dataset}"
        if not os.path.exists(kv_out_dir):
          os.mkdir(kv_out_dir)

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
                        max_gen, prompt_format, dataset, device, model_name, model2path, kv_out_dir))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()