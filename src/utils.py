#2. src/utils.py
#This handles your data ingestion. Run this script once to process data/corpus.txt into data/train.bin.

#Python

import torch
import numpy as np
from transformers import AutoTokenizer
import os

def prepare_data(input_path='data/corpus.txt', output_path='data/train.bin'):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if not os.path.exists(input_path):
        # Create a dummy file if it doesn't exist for testing
        with open(input_path, 'w') as f: f.write("Hello world! This is my first SLM training data.")
        
    with open(input_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    ids = tokenizer.encode(data)
    ids_array = np.array(ids, dtype=np.uint16)
    ids_array.tofile(output_path)
    print(f"Tokenized {len(ids)} tokens to {output_path}")

def get_batch(batch_size, block_size, device='cpu'):
    data = np.memmap('data/train.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    prepare_data()