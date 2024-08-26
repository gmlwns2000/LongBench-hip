import json
import argparse
import os

def parse_args(args=None):
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    
    with open('./config/model2path.json', 'r') as f:
        data = json.load(f)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=data.keys())
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--stride', type=int, default=None)
    parser.add_argument('--name', type=str, default='default')
    return parser.parse_args(args)