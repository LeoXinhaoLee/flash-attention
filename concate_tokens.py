import pdb
import os
import os.path as osp
from pathlib import Path
import glob
import random
import numpy as np


def sample_from_list(original_list, ratio):
    sample_size = max(1, int(len(original_list) * ratio))
    sampled_items = random.sample(original_list, sample_size)
    return sampled_items

random.seed(42)
np.random.seed(42)

directory = Path("/juice5/scr5/nlp/mttt/datasets/SlimPajama-627B-llama3-tokenized")
all_npy_files = directory.rglob("*.npy")

## @xinhao: mix ratio for 100B tokens used for 128K retrofit
# keywd = '500K_1M'
# keywd = 'above_1M'
# sample_ratio = 1.
# keywd = '10K_100K'
# sample_ratio = 0.25

## @xinhao: mix ratio for 100B tokens used for 8K retrofit
keywd = '10K_100K'
sample_ratio = 0.25


print(f'keyword: {keywd}')
print(f'sample ratio: {sample_ratio:.1%}')

filtered_files = [f for f in all_npy_files if keywd in f.name]
if sample_ratio < 1.:
    filtered_files = sample_from_list(filtered_files, sample_ratio)

data_all = []

for npy_file in filtered_files:
    data = np.load(npy_file, mmap_mode='r')
    data_all.append(data)

data_all = np.concatenate(data_all)
token_count = len(data_all)

print(f'Total token count: {token_count}')

output_path = f'/juice5/scr5/nlp/mttt/datasets/SlimPajama-627B-llama3-tokenized-100K-500K/{keywd}_tok_{token_count // (1024**3)}B_train.npy'
print('Saving...')
np.save(output_path, data_all)
