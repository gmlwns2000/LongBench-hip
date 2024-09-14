from datasets import load_dataset
import transformers
import numpy as np
import json

def sample():
    datasets = [
        'narrativeqa', 'qasper',
        'hotpotqa', '2wikimqa',
        'gov_report', 'multi_news',
    ]

    tokenizer = transformers.AutoTokenizer.from_pretrained('togethercomputer/LLaMA-2-7B-32K')

    result = {}
    for dataset in datasets:
        data = load_dataset('THUDM/LongBench', dataset, split='test')
        tls = []
        for sample in data:
            text = sample['input'] + sample['context'] + sample['answers'][0]
            token_length = len(tokenizer(text).input_ids)
            tls.append(token_length)
        tls = np.array(tls)
        tls_mean = np.mean(tls)
        tls_std = np.std(tls)
        tls = tls.tolist()
        result[dataset] = {
            'mean': tls_mean,
            'std': tls_std,
            'samples': tls,
        }

    path = 'pred/tokens.json'
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print('saved', path)

def plot():
    path = 'pred/tokens.json'

if __name__ == '__main__':
    # sample()
    plot()