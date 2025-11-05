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
from typing import Optional
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
    parser.add_argument('--sender_model', type=str.lower, default=None)
    parser.add_argument('--use_tune_cache', action='store_true', help="Whether to use tune cache")
    parser.add_argument('--dump_qkv', action='store_true', help="Whether to dump kv cache")
    parser.add_argument('--dataset', type=str.lower, nargs='+', default=["2wikimqa"], help="Specify one or more datasets to evaluate")
    parser.add_argument('--long_bench_ds_path', type=str, default="THUDM/LongBench", help="Path to the LongBench dataset if you have it locally")
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
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

def get_pred(
    use_tune_cache: bool,
    data: list,
    max_length: int,
    max_gen: int,
    prompt_format: str,
    dataset: str,
    device: torch.device,
    model_name: str,
    sender_model_name: Optional[str],
    model2path: dict,
    pred_out_path: str,
    dump_qkv: bool,
    qkv_out_dir: Optional[str] = None,
):

    """
    Run prediction for a subset of data across multiple ranks.

    Args:
        use_tune_cache (bool): Whether to enable tuned cache reuse.
        data (list): The dataset subset assigned to this rank.
        max_length (int): Maximum input length.
        max_gen (int): Maximum generation length.
        prompt_format (str): Prompt template format.
        dataset (str): Dataset name or identifier.
        device (torch.device): CUDA or CPU device.
        model_name (str): Model identifier.
        sender_model_name (str): Secondary model used for communication.
        model2path (dict): Mapping from model name to checkpoint path.
        pred_out_path (str): Output path for prediction results.
        dump_qkv (bool): Whether to dump KV cache.
        kv_out_dir (str): Directory for KV cache outputs.
    """
    device = torch.device(f'cuda:0')
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    model.config.architectures[0]
    sender_model = None
    sender_tokenizer = None
    if use_tune_cache:
        assert sender_model_name is not None, "Please specify sender_model when using tune cache."
        # 默认使用两张卡，卡1 跑 sender model，卡0 跑 receiver model
        assert torch.cuda.device_count() >= 2, "At least 2 GPUs are required when using tune cache."
        sender_device = torch.device(f'cuda:1')
        sender_model, sender_tokenizer = load_model_and_tokenizer(model2path[sender_model_name], sender_model_name, sender_device)
        assert sender_model.config.architectures[0] == model.config.architectures[0], f"Model architecture mismatch: {sender_model.config.architectures[0]} vs {model.config.architectures[0]}"
        assert type(sender_tokenizer) == type(tokenizer), f"Tokenizer type mismatch: {type(sender_tokenizer)} vs {type(tokenizer)}"
        
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        prompt = tokenizer.apply_chat_template([
            {"role": "user", "content": prompt},
        ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)

        context_length = input.input_ids.shape[-1]

        generate_args = dict(
            max_new_tokens=max_gen,
            do_sample=False,
            temperature=None,
            min_length=context_length + 1,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.encode("\n", add_special_tokens=False)[-1],
            ],
        )

        # generate sender model's kvcache
        if use_tune_cache:
            recompute_metadata = {
                "use_TuneCache": True,
                "kv_done": False,
            }
            sender_model.recompute_metadata = recompute_metadata
            sender_input = input.copy().to("cpu").to(sender_device)
            _ = sender_model.generate(**(sender_input),**generate_args)[0]
            llm_layers = sender_model.model.layers if hasattr(sender_model, 'model') else sender_model.layers

            reuse_layers = [0] * len(llm_layers)
            # 先尝试单层复用
            reuse_layers[19] = 1

            sender_cache = [[None, None, None] for _ in range(len(llm_layers))]
            for j in range(len(llm_layers)):
                if hasattr(llm_layers[j], 'self_attn') and hasattr(llm_layers[j].self_attn, 'hack_qkv') and reuse_layers[j] == 1:
                    hack_qkv = llm_layers[j].self_attn.hack_qkv
                    sender_cache[j][0] = hack_qkv[0].to("cpu").to(device)
                    sender_cache[j][1] = hack_qkv[1].to("cpu").to(device)
                    sender_cache[j][2] = hack_qkv[2].to("cpu").to(device)
                    torch.cuda.synchronize(device)
                else:
                    del llm_layers[j].self_attn.hack_qkv

        if use_tune_cache:
            recompute_metadata = {
                "use_TuneCache": True,
                "kv_done": True,
                "reuse_layers": reuse_layers,
            }
            model.recompute_metadata = recompute_metadata
            model.old_qkvs = sender_cache

        output = model.generate(**input, **generate_args)[0]

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(pred_out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

        if dump_qkv:
            hash_key = hash_prompt_sha256(prompt, bits=256, encoding="hex")
            llm_layers = model.model.layers if hasattr(model, 'model') else model.layers
            os.makedirs(qkv_out_dir,exist_ok=True)
            for j in range(len(llm_layers)):
                if hasattr(llm_layers[j], 'self_attn') and hasattr(llm_layers[j].self_attn, 'hack_qkv'):
                    hack_qkv = llm_layers[j].self_attn.hack_qkv
                    key_out_dir = os.path.join(qkv_out_dir, f"{hash_key}")
                    os.makedirs(key_out_dir, exist_ok=True)
                    os.makedirs(os.path.join(key_out_dir, f"layer_{j}"), exist_ok=True)  # create layer_j
                    torch.save(hack_qkv[0].cpu(), os.path.join(key_out_dir, f"layer_{j}", f"query.pt"))
                    torch.save(hack_qkv[1].cpu(), os.path.join(key_out_dir, f"layer_{j}", f"key.pt"))
                    torch.save(hack_qkv[2].cpu(), os.path.join(key_out_dir, f"layer_{j}", f"value.pt"))

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
    elif "qwen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    print(args)
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    use_tune_cache = args.use_tune_cache
    model_name = args.model
    sender_model_name = args.sender_model
    datasets = args.dataset
    long_bench_ds_path = args.long_bench_ds_path
    dump_qkv = args.dump_qkv

    if use_tune_cache and dump_qkv:
        raise ValueError("Cannot use tune cache and dump kv at the same time.")
    if use_tune_cache and (sender_model_name is None or model_name is None):
        raise ValueError("Please specify both sender_model and model when using tune cache.")

    # args.dataset can be a list of dataset names
    # define your model
    max_length = model2maxlen[model_name]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        data = load_dataset(long_bench_ds_path, dataset, split='test')
        print("load_dataset success")
        if not os.path.exists(f"pred/{model_name}"):
            os.makedirs(f"pred/{model_name}")
        pred_out_path = f"pred/{model_name}/{dataset}.jsonl"

        qkv_out_dir = None
        if dump_qkv:
            if not os.path.exists(f"kvcache/{model_name}"):
                os.makedirs(f"kvcache/{model_name}")
            qkv_out_dir = f"kvcache/{model_name}/{dataset}"
            if not os.path.exists(qkv_out_dir):
                os.mkdir(qkv_out_dir)

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        p = mp.Process(
            target=get_pred,
            args=(
                use_tune_cache,     # use_tune_cache
                data_all,           # data
                max_length,         # max_length
                max_gen,            # max_gen
                prompt_format,      # prompt_format
                dataset,            # dataset
                device,             # device
                model_name,         # model_name
                sender_model_name,  # sender_model_name
                model2path,         # model2path
                pred_out_path,      # out_path
                dump_qkv,           # dump_qkv
                qkv_out_dir,        # kv_out_dir
            ),
        )

        p.start()
        p.join()