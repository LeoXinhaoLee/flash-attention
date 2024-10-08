import os
import pdb
from pathlib import Path
current_dir = Path(__file__).parent.absolute()

import pytest
import argparse
import shutil
import subprocess

import torch
import numpy as np

import dotenv

from training.src.datamodules.language_modeling_SPJ_hf import LMDataModule
# from training.src.datamodules.language_modeling_hf import LMDataModule


def div_up(x: int, y: int) -> int:
    return (x + y - 1) // y


# https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python/55423170#55423170
def num_cpu_cores():
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except ImportError:
        return len(os.sched_getaffinity(0))


def main(args):
    chunk_name = args.chunk_name

    data_path_prefix = '/juice5/scr5/nlp/mttt/datasets'

    hf_cache = f'/juice5/scr5/nlp/mttt/hf_hub_tmp_test/{chunk_name}'
    os.environ['HF_HOME'] = hf_cache
    if os.path.isdir(hf_cache):
        shutil.rmtree(hf_cache, ignore_errors=True)
    os.makedirs(hf_cache)

    dataset_name = os.path.join(data_path_prefix, f'SlimPajama-627B-tmp-test/{chunk_name}')
    os.makedirs(dataset_name, exist_ok=True)
    subprocess.run(['cp', '-r', os.path.join(data_path_prefix, f'SlimPajama-627B-tmp-test/test'), dataset_name], check=True)
    subprocess.run(['cp', '-r', os.path.join(data_path_prefix, f'SlimPajama-627B-tmp-test/validation'), dataset_name], check=True)

    original_dataset_path = os.path.join(data_path_prefix, 'SlimPajama-627B-test')
    original_chunk_path = os.path.join(original_dataset_path, 'train', chunk_name)  # e.g., chunk1, diff chunk parallelized by diff machines

    os.makedirs(os.path.join(data_path_prefix, 'SlimPajama-627B-llama3-tokenized-test'), exist_ok=True)
    chunk_cache_path = os.path.join(data_path_prefix, 'SlimPajama-627B-llama3-tokenized-test', chunk_name)  # save .npy
    os.makedirs(chunk_cache_path, exist_ok=True)

    file_list = os.listdir(original_chunk_path)
    file_list.sort()

    files_per_iteration = 2
    n_iter = len(file_list) // files_per_iteration
    file_splits = np.array_split(file_list, n_iter)

    first_part_id_to_write = 0
    if args.resume:
        existing_part_ids = os.listdir(chunk_cache_path)
        existing_last_part_id = len(existing_part_ids) - 1  # overwrite the latest part as it might be half-complete
        if existing_last_part_id >= 0:
            if os.path.exists(os.path.join(chunk_cache_path, f'part_{existing_last_part_id}', 'SUCCESS')):
                existing_last_part_id += 1  # don't overwrite the last part
        first_part_id_to_write = max(first_part_id_to_write, existing_last_part_id)
        if os.path.exists(os.path.join(chunk_cache_path, f'part_{first_part_id_to_write}')):
            shutil.rmtree(os.path.join(chunk_cache_path, f'part_{first_part_id_to_write}'), ignore_errors=True)

    print(f'Starting at chunk {chunk_name} part {first_part_id_to_write}')

    for part_id, cur_iter_files in enumerate(file_splits):
        if part_id < first_part_id_to_write:
            continue

        shutil.rmtree(os.path.join(dataset_name, 'train'), ignore_errors=True)
        os.makedirs(os.path.join(dataset_name, 'train'), exist_ok=True)
        for file in cur_iter_files:
            s_file_path = os.path.join(original_dataset_path, 'train', chunk_name, file)
            t_file_path = os.path.join(dataset_name, 'train', file)
            subprocess.run(['ln', s_file_path, t_file_path], check=True)

        cache_dir = os.path.join(chunk_cache_path, f'part_{part_id}')  # path to save tokenized dataset of current part id
        num_workers = num_cpu_cores() // 2
        # num_workers = 1
        chunk_size = 16
        datamodule = LMDataModule(dataset_name, tokenizer_name='meta-llama/Meta-Llama-3.1-8B',
                                  dataset_config_name=None,
                                  max_length=2048, cache_dir=Path(cache_dir),
                                  add_eos=True, batch_size=8,
                                  num_workers=num_workers, use_shmem=False,
                                  raw_json_path=None, pad_to_multiple_of=chunk_size)
        datamodule.prepare_data()

        shutil.rmtree(os.path.join(dataset_name, 'train'), ignore_errors=True)
        f = open(os.path.join(cache_dir, 'SUCCESS'), 'w')
        f.close()
        if os.path.isdir(hf_cache):
            shutil.rmtree(hf_cache, ignore_errors=True)
        os.makedirs(hf_cache)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tokenizing SlimPajama in Single Process with Flash-attention Script")
    parser.add_argument("--chunk_name", type=str, default='chunk10')
    parser.add_argument("--resume", action='store_true')
    args = parser.parse_args()

    main(args)


