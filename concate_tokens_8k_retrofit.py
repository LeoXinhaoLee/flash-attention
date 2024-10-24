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


def sample_from_list(original_list, ratio, skip=0):
    sample_size = max(1, int(len(original_list) * ratio))
    sampled_items = random.sample(original_list, sample_size)
    while skip > 0:
        original_list = list(set(original_list) - set(sampled_items))
        sampled_items = random.sample(original_list, sample_size)
        skip -= 1

    return sampled_items

random.seed(42)
np.random.seed(42)

directory = Path("/juice5/scr5/nlp/mttt/datasets/SlimPajama-627B-llama3-tokenized")
all_npy_files = directory.rglob("*.npy")

output_dir = '/juice5/scr5/nlp/mttt/datasets/SlimPajama-627B-llama3-tokenized-100B-below-100K'
os.makedirs(output_dir, exist_ok=True)
output_file = osp.join(output_dir, 'train.h5')

## @xinhao: mix ratio for 100B tokens used for 8K retrofit
output_name = 'below_100K'
config_list = [
    # dict(keywd='10K_100K',
    #      sample_ratio=0.25,  # 50B
    #      skip=1,
    #      batch_size=10),
    dict(keywd='below_10K',
         sample_ratio=0.06,  # 50B
         skip=0,
         batch_size=2)
]

with h5py.File(output_file, 'a') as f_out:
    if 'train_dataset' not in f_out:
        dset = f_out.create_dataset('train_dataset', shape=(0,), maxshape=(None,), dtype='int32')
    else:
        dset = f_out['train_dataset']

    for config in config_list:
        keywd = config['keywd']
        sample_ratio = config['sample_ratio']
        skip = config.get('skip', 0)
        batch_size = config.get('batch_size', 1)
        print(f'keyword: {keywd}')
        print(f'sample ratio: {sample_ratio:.1%}')

        filtered_files = [f for f in all_npy_files if keywd in f.name]
        if sample_ratio < 1.:
            filtered_files = sample_from_list(filtered_files, sample_ratio, skip)

        token_count = 0
        batch = []
        for npy_file in tqdm(filtered_files, desc=f"Appending {keywd}"):
            # data = np.load(npy_file, mmap_mode='r')
            data = np.load(npy_file)
            batch.append(data)
            token_count += data.shape[0]

            if len(batch) >= batch_size:
                concatenated_batch = np.concatenate(batch, axis=0)
                dset.resize(dset.shape[0] + concatenated_batch.shape[0], axis=0)
                dset[-concatenated_batch.shape[0]:] = concatenated_batch
                batch.clear()
                gc.collect()

        if batch:
            concatenated_batch = np.concatenate(batch, axis=0)
            dset.resize(dset.shape[0] + concatenated_batch.shape[0], axis=0)
            dset[-concatenated_batch.shape[0]:] = concatenated_batch
            batch.clear()
        gc.collect()

        print(f'Total token count in {keywd}: {token_count}')

    print(f'Total token count: {dset.shape[0]}')
    print(f'Data saved to {output_file}')
