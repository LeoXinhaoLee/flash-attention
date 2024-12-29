import pdb
import os
import os.path as osp
from pathlib import Path
import glob
import random
from tqdm import tqdm
import gc
import numpy as np
import h5py


all_npy_files = [
    '/home/xinhaoli/datasets/DCLM-Baseline-100B-llama3-tokenized-no-pad/tokenizer_name-meta-llama/Meta-Llama-3.1-8B-val_ratio-0.0005-val_split_seed-2357-add_eos-True-detokenize-False/train.npy',
    '/home/xinhaoli/datasets/DCLM-Baseline-25B-llama3-tokenized-no-pad/tokenizer_name-meta-llama/Meta-Llama-3.1-8B-val_ratio-0.0005-val_split_seed-2357-add_eos-True-detokenize-False/train.npy',
    # '/home/xinhaoli/datasets/DCLM-Baseline-100B-llama3-tokenized-no-pad/tokenizer_name-meta-llama/Meta-Llama-3.1-8B-val_ratio-0.0005-val_split_seed-2357-add_eos-True-detokenize-False/test.npy',
    # '/home/xinhaoli/datasets/DCLM-Baseline-25B-llama3-tokenized-no-pad/tokenizer_name-meta-llama/Meta-Llama-3.1-8B-val_ratio-0.0005-val_split_seed-2357-add_eos-True-detokenize-False/test.npy',
]

output_dir = '/home/xinhaoli/datasets/DCLM-Baseline-108B-llama3-tokenized-no-pad-h5/tokenizer_name-meta-llama/Meta-Llama-3.1-8B-val_ratio-0.0005-val_split_seed-2357-add_eos-True-detokenize-False'
os.makedirs(output_dir, exist_ok=True)
output_file = osp.join(output_dir, 'train.h5')


chunk_size = 268435456 * 8  # this number of int32 -> 1GB * 4 = 8GB

with h5py.File(output_file, 'a') as f_out:
    if 'train_dataset' not in f_out:
        dset = f_out.create_dataset('train_dataset', shape=(0,), maxshape=(None,), dtype='int32')
    else:
        dset = f_out['train_dataset']

    print(f'Existing token: {dset.shape[0]}')

    token_count = 0
    for npy_file in tqdm(all_npy_files, desc="Appending npy file"):
        data = np.load(npy_file, mmap_mode='r')
        total_size = data.shape[0]

        # Process the data in chunks
        for start in tqdm(range(0, total_size, chunk_size), "Processing chunks"):
            end = min(start + chunk_size, total_size)
            chunk = data[start:end]  # Only this chunk is loaded into memory

            # Resize and append chunk
            dset.resize(dset.shape[0] + chunk.shape[0], axis=0)
            dset[-chunk.shape[0]:] = chunk
            token_count += chunk.shape[0]

            # Ensure the chunk is released from memory immediately
            del chunk
            gc.collect()

        # Ensure memory from current npy file is released
        del data  # delete memory-mapped object
        gc.collect()  # force garbage collection to free memory

# print(f'Total token count: {dset.shape[0]}')
print(f'Add token: {token_count}')
print(f'Data saved to {output_file}')
