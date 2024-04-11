import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm import LLM, SamplingParams

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=[
        "llama2-7b-chat-4k", 
        "longchat-v1.5-7b-32k", 
        "xgen-7b-8k", 
        "internlm-7b-8k", 
        "chatglm2-6b", 
        "chatglm2-6b-32k", 
        "chatglm3-6b-32k", 
        "vicuna-v1.5-7b-16k",
        "llama2-7b-chat-32k",
    ])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--stride', type=int, default=None)
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
    elif "llama2" in model_name:
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

ATTENTION_METHOD = os.getenv('ATTENTION_METHOD', 'none')
HIP_K = int(os.getenv('HIP_K', '512'))

def get_pred(
    rank, 
    world_size, 
    data, 
    max_length, 
    max_gen, 
    prompt_format, 
    dataset, 
    device, 
    model_name, 
    model2path, 
    out_path,
    model=None,
    tokenizer=None,
):
    device = torch.device(f'cuda:{rank}')
    if model is None and tokenizer is None:
        raise Exception()
        model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    
    with open(out_path, "w", encoding="utf-8") as f:
        for json_obj in tqdm(data):
            prompt = prompt_format.format(**json_obj)
            # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if "chatglm3" in model_name:
                tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
                prompt = build_chat(tokenizer, prompt, model_name)
            if "chatglm3" in model_name:
                if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                    input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
                else:
                    input = prompt.to(device)
            else:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input.input_ids.shape[-1]
            
            if ATTENTION_METHOD == 'streaming_llm':
                if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
                    raise Exception()
                    with torch.inference_mode():
                        output = model.generate(
                            **input,
                            max_new_tokens=max_gen,
                            num_beams=1,
                            do_sample=False,
                            temperature=1.0,
                            min_length=context_length+1,
                            eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                        )[0]
                else:
                    output = model.generate(
                        **input,
                        max_new_tokens=max_gen,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                    )[0]
                pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            else:
                sampling_params = SamplingParams(
                    temperature=1.0,
                    top_p=1.0,
                    top_k=1, # No sampleing
                    max_tokens=max_gen,
                    frequency_penalty=0.0,
                    repetition_penalty=1.0,
                    ignore_eos=False,
                    skip_special_tokens=True,
                )
                
                prompt = tokenizer.decode(input.input_ids[0], skip_special_tokens=True)
                vllm_outputs = model.generate(
                    prompt, 
                    sampling_params,
                    use_tqdm=False,
                )
                pred = vllm_outputs[0].outputs[0].text
            
            pred = post_process(pred, model_name)
            
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
            f.flush()
    # dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device, seq_len):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    
    if ATTENTION_METHOD == 'streaming_llm':
        from timber.models.modeling_llama import LlamaCustomAttention
        
        config = AutoConfig.from_pretrained(path)
        config.attn_implementation = config._attn_implementation = 'sdpa'
        model = LlamaForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            device_map={'':device}
        )
        
        for m in model.modules():
            if isinstance(m, LlamaCustomAttention):
                m.attention_method = 'streaming_llm'
                m.tree_k = HIP_K
        
        model.eval()
    else:
        model = LLM(
            path,
            max_num_seqs=1,
            max_context_len_to_capture=seq_len + 500,
            max_model_len=seq_len + 500,
            swap_space=0,
            kv_cache_dtype='fp8_e5m2',
            dtype='half',
            gpu_memory_utilization=0.9,
            tensor_parallel_size=torch.cuda.device_count(),
            enforce_eager=os.environ.get('FORCE_EAGER','0')=='1',
            trust_remote_code=True,
        )
    
    return model, tokenizer
    
    # if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
    #     tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    #     model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    # elif "llama2" in model_name:
    #     # replace_llama_attn_with_flash_attn()
    #     tokenizer = LlamaTokenizer.from_pretrained(path)
    #     config = AutoConfig.from_pretrained(path)
    #     config.attn_implementation = config._attn_implementation = 'flash_attention_2'
    #     model = LlamaForCausalLM.from_pretrained(
    #         path,
    #         config=config,
    #         torch_dtype=torch.bfloat16,
    #         load_in_4bit=True,
    #         device_map={'':device}
    #     )#.to(device)
    # elif "longchat" in model_name or "vicuna" in model_name:
    #     from fastchat.model import load_model
    #     replace_llama_attn_with_flash_attn()
    #     model, _ = load_model(
    #         path,
    #         device='cpu',
    #         num_gpus=0,
    #         load_8bit=False,
    #         load_4bit=True,
    #         cpu_offloading=False,
    #         debug=False,
    #     )
    #     model = model.to(device)
    #     model = model.bfloat16()
    #     tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    # model = model.eval()
    # return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = 1
    # mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name] if args.stride is None else args.stride
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
        #             "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
        #             "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
        datasets = [
            'narrativeqa', 'qasper',
            'hotpotqa', '2wikimqa',
            'gov_report', 'multi_news',
        ]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    
    model, tokenizer = load_model_and_tokenizer(
        model2path[model_name],
        model_name,
        device,
        max_length,
    )
    
    for dataset in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            out_path = f"pred/{model_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            get_pred(
                rank, 
                world_size, 
                data_subsets[rank], 
                max_length,
                max_gen, 
                prompt_format, 
                dataset, 
                device, 
                model_name, 
                model2path, 
                out_path,
                model=model,
                tokenizer=tokenizer,
            )