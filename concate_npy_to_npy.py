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


random.seed(42)
np.random.seed(42)

directory = Path("/juice5/scr5/nlp/mttt/datasets/SlimPajama-627B-llama3-tokenized")
all_npy_files = list(directory.rglob("*.npy"))
all_npy_files.sort()
random.shuffle(all_npy_files)

## 10/24 night
# st = 0
# ed = 500

## 10/25 day 1
# st = 500
# ed = 1000

# ## 10/25 day 2
# st = 1000
# ed = 1500
# print(f'St: {st}, Ed: {ed}')
# all_npy_files = all_npy_files[st:ed]

## 10/25 day 3
st = 1500
ed = len(all_npy_files)

print(f'St: {st}, Ed: {ed}')
all_npy_files = all_npy_files[st:ed]

output_dir = '/juice5/scr5/nlp/mttt/datasets/SlimPajama-627B-llama3-tokenized-cat-all'
os.makedirs(output_dir, exist_ok=True)
output_file = osp.join(output_dir, 'train.h5')


chunk_size = 268435456  # this number of int32 -> 1GB

with h5py.File(output_file, 'a') as f_out:
    if 'train_dataset' not in f_out:
        dset = f_out.create_dataset('train_dataset', shape=(0,), maxshape=(None,), dtype='int32')
    else:
        dset = f_out['train_dataset']

    print(f'Existing token: {dset.shape[0]}')

    token_count = 0
    for npy_file in tqdm(all_npy_files, desc="Appending npy file"):
        data = np.memmap(npy_file, dtype='int32', mode='r')
        total_size = data.shape[0]

        # Process the data in chunks
        for start in range(0, total_size, chunk_size):
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
