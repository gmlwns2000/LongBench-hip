import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import numpy as np
import random
import torch.distributed as dist
import torch.multiprocessing as mp

from hip.models.modeling_llama import LlamaForCausalLM
from hip.models.qwen.modeling_qwen2 import Qwen2ForCausalLM

from vllm import LLM, SamplingParams

from args import parse_args

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
        prompt = f"[INST]\n{prompt}\n[/INST]\n\n"
    elif "llama3.1" in model_name:
        if "pretrained" in model_name:
            f"""-------------------------------------------------------------------------------
[System]: You are helpful assistant.
-------------------------------------------------------------------------------
[User]: Hi! I want to give a task about following document. Here is the context.

{prompt}

Now, please answer given task.
-------------------------------------------------------------------------------
[Assistant]: Sure! Here is my response. Response: """
        else:
            prompt = f"""<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "qwen2" in model_name:
        prompt = f'<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'
    elif "llama3" in model_name:
        prompt = f'<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    elif "phi3" in model_name:
        raise Exception('phi3 not supported yet on vllm-hip')
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

ATTENTION_METHOD = os.getenv('ATTENTION_METHOD', 'none')
HIP_K = int(os.getenv('HIP_K', '512'))
import transformers

class StoppingCriteriaSub(transformers.StoppingCriteria):
    def __init__(self, stops = [], tokenizer = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if self.tokenizer.decode(stop) == self.tokenizer.decode(last_token):
                return True
        return False

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
        for json_obj in tqdm(data, desc=dataset):
            prompt = prompt_format.format(**json_obj)
            # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if "chatglm3" in model_name:
                tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=False)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=False)
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
            
            if ATTENTION_METHOD in ['streaming_llm', 'h2o']:
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
                    stop_words = ["<|eot_id|>"]
                    stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
                    stopping_criteria = transformers.StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, tokenizer=tokenizer)])
                    
                    with torch.inference_mode():
                        output = model.generate(
                            **input,
                            max_new_tokens=max_gen,
                            num_beams=1,
                            do_sample=False,
                            temperature=1.0,
                            stopping_criteria=stopping_criteria,
                        )[0]
                for m in model.modules():
                    if hasattr(m, '_clean_cache'):
                        m._clean_cache()
                pred = tokenizer.decode(output[context_length:], skip_special_tokens=False)
            else:
                stop = []
                if 'llama3' in model_name:
                    stop.append('<|eot_id|>')
                sampling_params = SamplingParams(
                    temperature=1.0,
                    top_p=1.0,
                    top_k=1, # No sampleing
                    max_tokens=max_gen,
                    frequency_penalty=0.0,
                    repetition_penalty=1.0,
                    ignore_eos=False,
                    skip_special_tokens=False,
                    stop=stop,
                )
                
                prompt = tokenizer.decode(input.input_ids[0], skip_special_tokens=False)
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
        from hip.models.modeling_llama import LlamaCustomAttention
        from hip.models.qwen.modeling_qwen2 import Qwen2CustomAttention
        
        config = AutoConfig.from_pretrained(path)
        config.attn_implementation = config._attn_implementation = 'sdpa'
        config.max_position_embeddings = 131072
        
        ModelClass = LlamaForCausalLM
        if 'qwen2' in model_name:
            ModelClass = Qwen2ForCausalLM
        
        model = ModelClass.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.bfloat16,
            # load_in_4bit=True,
            device_map={'':device}
        )
        
        num_patched = 0
        for m in model.modules():
            if isinstance(m, (LlamaCustomAttention, Qwen2CustomAttention)):
                assert hasattr(m, 'attention_method')
                m.attention_method = 'streaming_llm'
                m.tree_k = HIP_K
                num_patched += 1
        assert num_patched > 0
        
        model.eval()
    elif ATTENTION_METHOD == 'h2o':
        from hip.models.modeling_llama import LlamaForCausalLM, LlamaConfig
        
        config = LlamaConfig.from_pretrained(path)
        config._attn_implementation = config.attn_implementation = 'sdpa'
        infer_dtype = torch.bfloat16
        ModelClass = LlamaForCausalLM
        
        config.hh_size = HIP_K // 2
        config.recent_size = HIP_K // 2
        config._attn_implementation = config.attn_implementation = 'eager'
        config.shift_q_pos = False
        config.streaming = False
        config.reduction_for_gqa = 'average'
        config.is_decoding = True # use dense prefill
        
        model = ModelClass.from_pretrained(
            path,
            config=config,
            device_map={'': device},
            quantization_config=None,
            torch_dtype=infer_dtype,
            # torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        for m in model.modules():
            if hasattr(m, 'attention_method'):
                m.attention_method = 'h2o'
                m.tree_k = HIP_K
                m.tree_block_size_q = 64
                m.tree_block_stride_q = 2
                m.tree_block_size_k = 2
                m.tree_block_stride_k = 1
                m.tree_using_context_avg = False
                m.tree_dense_queries = -1
                m.tree_dense_layers = list(range(3))
                m.tree_rope_method = 'none'
                m.tree_enable_sparq = False
                m.tree_enable_flash = True
                m.tree_use_sliding_window = True
                m.tree_sampling_method = 'center'
            for m in model.modules():
                if hasattr(m, 'attention_method'):
                    m.tree_using_context_avg = False
        model = model.eval()
    else: 
        model = LLM(
            path,
            max_num_seqs=1,
            max_seq_len_to_capture=seq_len + 512,
            max_model_len=seq_len + 512,
            swap_space=0,
            kv_cache_dtype=os.getenv('KV_CACHE_DTYPE', 'auto'),
            dtype='half',
            gpu_memory_utilization=float(os.getenv('MEM_UTIL', '0.9')),
            tensor_parallel_size=torch.cuda.device_count(),
            enforce_eager=os.environ.get('ENFORCE_EAGER','0')=='1',
            trust_remote_code=True,
            enable_chunked_prefill=False,
            max_num_batched_tokens=seq_len + 512,
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
    
    # vllm will parallelize
    world_size = 1
    mp.set_start_method('spawn', force=True)

    model_name = args.model
    model2path = json.load(open("config/model2path.json", "r"))
    if os.getenv('OVERRIDE_MODEL_PATH', '') != '':
        model2path[model_name] = os.getenv('OVERRIDE_MODEL_PATH', '')
        print('Using', model_name, model2path[model_name])
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    os.makedirs("pred", exist_ok=True)
    os.makedirs("pred_e", exist_ok=True)
    
    model, tokenizer = load_model_and_tokenizer(
        model2path[model_name],
        model_name,
        device,
        max_length,
    )
    
    for dataset in datasets:
        pred_root_name = None
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{args.name}/{model_name}"):
                os.makedirs(f"pred_e/{args.name}/{model_name}")
            out_path = f"pred_e/{args.name}/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"pred/{args.name}/{model_name}"):
                os.makedirs(f"pred/{args.name}/{model_name}")
            out_path = f"pred/{args.name}/{model_name}/{dataset}.jsonl"
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