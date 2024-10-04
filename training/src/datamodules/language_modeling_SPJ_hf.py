import pdb

# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
from itertools import chain
from pathlib import Path
import pickle
from typing import Any, List, Union
import subprocess
import mmap
from collections import defaultdict
import multiprocessing
from tqdm import tqdm
import pandas as pd
import time

from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Pool

import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from datasets import Dataset as HF_Dataset

from pytorch_lightning import LightningDataModule

from src.datamodules.datasets.lm_dataset import LMDataset
from src.datamodules.fault_tolerant_sampler import RandomFaultTolerantSampler
from src.datamodules.fault_tolerant_sampler import FaultTolerantDistributedSampler
from src.datamodules.datasets.detokenizer import DATASET_TOKENIZATION_REGISTRY
# from src.utils.utils import get_logger
# logger = get_logger()


# Move the function outside of the class
def tokenize_and_categorize(example, tokenizer, add_eos, pad_to_multiple_of, meta_column_name):
    def tokenize_fn(text):
        if add_eos:
            text = text + tokenizer.eos_token
        return tokenizer(text)

    if pad_to_multiple_of > 0:
        _tokenize_fn = tokenize_fn

        def pad_to_multiple(tokens, pad_token_id, multiple):
            length = len(tokens)
            padding_length = (multiple - length % multiple) % multiple
            return tokens + [pad_token_id] * padding_length

        def tokenize_pad_to_multiple(text):
            tokenize_dic = _tokenize_fn(text)
            input_ids = tokenize_dic.pop('input_ids')
            input_ids = pad_to_multiple(input_ids, tokenizer.bos_token_id, pad_to_multiple_of)
            return {'input_ids': input_ids, **tokenize_dic}

        tokenize_fn = tokenize_pad_to_multiple

    # Tokenize the text and categorize based on its length
    domain = example[meta_column_name]['redpajama_set_name']
    tokenized = tokenize_fn(example['text'])  # Assuming 'text' is the key for text data
    input_ids = tokenized['input_ids']
    length = len(input_ids)

    # Classify based on length ranges
    if length < 10 * 1e3:
        range_key = 'tok_len_below_10K'
    elif length < 100 * 1e3:
        range_key = 'tok_len_10K_100K'
    elif length < 200 * 1e3:
        range_key = 'tok_len_100K_200K'
    elif length < 500 * 1e3:
        range_key = 'tok_len_200K_500K'
    elif length < 1000 * 1e3:
        range_key = 'tok_len_500K_1M'
    else:
        range_key = 'tok_len_above_1M'

    # Return the categorized result
    return {'domain': domain, 'range_key': range_key, 'input_ids': input_ids, 'len': length}


# This function replaces the lambda function used in pool.map
def tokenize_and_categorize_wrapper(args):
    return tokenize_and_categorize(*args)


class LMDataModule(LightningDataModule):
    def __init__(self, dataset_name, tokenizer_name, dataset_config_name=None, max_length=1024,
                 cache_dir=None, val_ratio=0.0005, val_split_seed=2357, add_eos=True,
                 detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                 use_shmem=True, raw_json_path=None, finetune_ratio=None, pad_to_multiple_of=0):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.max_length = max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

        self.use_shmem = use_shmem
        if self.use_shmem:
            assert cache_dir is not None

        # @xinhao: add option to specify raw json path directly
        self.raw_json_path = raw_json_path
        self.finetune_ratio = finetune_ratio

        # @xinhao: add option for end-document padding
        self.pad_to_multiple_of= pad_to_multiple_of

    def prepare_data(self):
        if self.cache_dir is None:  # Just download the dataset
            load_dataset(self.dataset_name, self.dataset_config_name)
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == 'test' and hasattr(self, 'dataset_test'):
            return

        concat_ids, self.tokenizer = self.process_dataset()

        self.vocab_size = len(self.tokenizer)

        # Create all splits
        if self.dataset_name == 'books3_splitted_finetune':
            self.dataset_train, self.dataset_finetune, \
            self.dataset_val, self.dataset_test = [
                LMDataset(concat_ids[split], seq_len=self.max_length)
                for split in ['train', 'finetune', 'validation', 'test']
            ]
        else:
            self.dataset_train, self.dataset_val, self.dataset_test = [
                LMDataset(concat_ids[split], seq_len=self.max_length)
                for split in ['train', 'validation', 'test']
            ]


    # def process_dataset(self):
    #     cache_dir = None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
    #     if cache_dir is not None:
    #         if cache_dir.is_dir():
    #             return self._load_from_cache(cache_dir)
    #
    #     print('Start Loading!')
    #     raw_datasets = load_dataset(self.dataset_name, self.dataset_config_name)
    #     print('Loading Done!')
    #
    #     tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
    #
    #     column_names = raw_datasets["train"].column_names  # 'text', 'meta'
    #     text_column_name = "text"
    #     meta_column_name = "meta"
    #
    #     assert self.add_eos
    #     dtype = np.uint16 if tokenizer.vocab_size < 64 * 1024 else np.int32
    #
    #     def tokenize_fn(text):
    #         if self.add_eos:
    #             text = text + tokenizer.eos_token
    #         return tokenizer(text)
    #
        # if self.pad_to_multiple_of > 0:
        #     _tokenize_fn = tokenize_fn
        #
        #     def pad_to_multiple(tokens, pad_token_id, multiple):
        #         length = len(tokens)
        #         padding_length = (multiple - length % multiple) % multiple
        #         return tokens + [pad_token_id] * padding_length
        #
        #     def tokenize_pad_to_multiple(example):
        #         tokenize_dic = _tokenize_fn(example)
        #         input_ids = tokenize_dic.pop('input_ids')
        #         input_ids = pad_to_multiple(input_ids, tokenizer.bos_token_id, self.pad_to_multiple_of)
        #         return {'input_ids': input_ids, **tokenize_dic}
    #
    #         tokenize_fn = tokenize_pad_to_multiple
    #
    #     def tokenize_concat(examples):
    #         input_ids = [tokenize_fn(seq)['input_ids'] for seq in examples[text_column_name]]
    #         lengths = [len(ids) for ids in input_ids]
    #         return {'input_ids': input_ids, 'len': lengths}
    #
    #     # Initialize a defaultdict to store grouped data by domain
    #     domain_datasets = defaultdict(list)
    #
    #     # Group examples by their 'meta' field in a single pass
    #     for example in raw_datasets['train']:
    #         domain = example[meta_column_name]['redpajama_set_name']
    #         domain_datasets[domain].append(example)
    #
    #     concat_ids = {}
    #     assert cache_dir is not None
    #     cache_dir.mkdir(parents=True, exist_ok=True)
    #
    #     # Process each domain's dataset
    #     for domain, examples in domain_datasets.items():
    #         # Create a Dataset from the list of examples
    #         domain_dataset = HF_Dataset.from_dict(
    #             {key: [example[key] for example in examples] for key in examples[0].keys()})
    #
    #         # Tokenize the examples and separate them by length (<1000 tokens vs >=1000 tokens)
    #         tokenized_datasets = domain_dataset.map(
    #             tokenize_concat,
    #             batched=True,
    #             num_proc=max(self.num_workers, 1),
    #             remove_columns=column_names,
    #             desc=f"Running tokenizer on {domain} dataset",
    #         )
    #
    #         # Separate examples into two groups based on their length
    #         domain_length_example_dict = {
    #             'tok_len_below_10K': tokenized_datasets.filter(lambda x: x['len'] < 10 * 1e3),
    #             'tok_len_10K_100K': tokenized_datasets.filter(lambda x: x['len'] >= 10 * 1e3 and x['len'] < 100 * 1e3),
    #             'tok_len_100K_200K': tokenized_datasets.filter(lambda x: x['len'] >= 100 * 1e3 and x['len'] < 200 * 1e3),
    #             'tok_len_200K_500K': tokenized_datasets.filter(lambda x: x['len'] >= 200 * 1e3 and x['len'] < 500 * 1e3),
    #             'tok_len_500K_1M': tokenized_datasets.filter(lambda x: x['len'] >= 500 * 1e3 and x['len'] < 1000 * 1e3),
    #             'tok_len_above_1M': tokenized_datasets.filter(lambda x: x['len'] >= 1000 * 1e3),
    #         }
    #
    #         def write_ids_to_disk(example, filename):
    #             with open(filename, 'r+b') as f:
    #                 mm = mmap.mmap(f.fileno(), 0)
    #                 start_idx = example['len_offset'] - len(example['input_ids'])
    #                 array_len = len(example['input_ids'])
    #                 arr = np.ndarray((array_len,), dtype=dtype, buffer=mm,
    #                                  offset=np.dtype(dtype).itemsize * start_idx)
    #                 arr[:] = example['input_ids']
    #                 mm.flush()
    #
    #         # Function to process a group (short or long) of examples
    #         def process_group(group_name, dataset):
    #             dataset = dataset.add_column(
    #                 'len_offset', np.cumsum(dataset['len'])
    #             )
    #             array_len = dataset[-1]['len_offset']
    #
    #             # Save the tokenized examples for this group (short/long) and domain
    #             filename = cache_dir / f'{domain}_{group_name}.bin'
    #             subprocess.run(['truncate', '-s', str(array_len * np.dtype(dtype).itemsize),
    #                             str(filename)], check=True)
    #
    #             dataset.map(
    #                 write_ids_to_disk,
    #                 fn_kwargs={'filename': filename},
    #                 batched=False,
    #                 num_proc=max(self.num_workers, 1),
    #                 desc=f"Concatenating {group_name} examples for {domain}",
    #             )
    #             concat_ids[f"{domain}_{group_name}"] = np.memmap(filename, dtype=dtype, mode='r', shape=(array_len,))
    #
    #         # Process short examples
    #         for len_range in domain_length_example_dict.keys():
    #             if len(domain_length_example_dict[len_range]['len']) > 0:
    #                 process_group(len_range, domain_length_example_dict[len_range])
    #
    #     if cache_dir is not None:
    #         self._save_to_cache(concat_ids, tokenizer, cache_dir)
    #         if not self.use_shmem:
    #             for domain_group in concat_ids:
    #                 Path(cache_dir / f'{domain_group}.bin').unlink()
    #
    #     return concat_ids, tokenizer

    # def process_dataset(self):
    #     cache_dir = None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
    #     if cache_dir is not None:
    #         if cache_dir.is_dir():
    #             return self._load_from_cache(cache_dir)
    #
    #     print('Start Loading!')
    #     raw_datasets = load_dataset(self.dataset_name, self.dataset_config_name)
    #     print('Loading Done!')
    #
    #     tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
    #
    #     column_names = raw_datasets["train"].column_names  # 'text', 'meta'
    #     text_column_name = "text"
    #     meta_column_name = "meta"
    #
    #     assert self.add_eos
    #     dtype = np.uint16 if tokenizer.vocab_size < 64 * 1024 else np.int32
    #
    #     # Using multiprocessing to parallelize categorization and tokenization
    #     def parallel_tokenize_and_categorize(dataset, num_proc):
    #         # Prepare arguments for the function
    #         args = [(example, tokenizer, self.add_eos, self.pad_to_multiple_of, meta_column_name) for example in dataset]
    #         # Use pool map with the regular function instead of a lambda
    #         with multiprocessing.Pool(processes=num_proc) as pool:
    #             results = pool.map(tokenize_and_categorize_wrapper, args)
    #         return results
    #
    #     # Tokenize and categorize the examples in parallel
    #     categorized_examples = parallel_tokenize_and_categorize(raw_datasets['train'], num_proc=self.num_workers)
    #
    #     # Initialize a defaultdict to store grouped data by domain and length range
    #     domain_length_example_dict = defaultdict(lambda: defaultdict(list))
    #
    #     # Group tokenized examples by domain and length range
    #     for result in categorized_examples:
    #         domain = result['domain']
    #         range_key = result['range_key']
    #         input_ids = result['input_ids']
    #         length = result['len']
    #
    #         # Store tokenized example directly in the grouped dictionary
    #         domain_length_example_dict[domain][range_key].append({'input_ids': input_ids, 'len': length})
    #
    #     concat_ids = {}
    #     assert cache_dir is not None
    #     cache_dir.mkdir(parents=True, exist_ok=True)
    #
    #     def write_ids_to_disk(example, filename):
    #         with open(filename, 'r+b') as f:
    #             mm = mmap.mmap(f.fileno(), 0)
    #             start_idx = example['len_offset'] - len(example['input_ids'])
    #             array_len = len(example['input_ids'])
    #             arr = np.ndarray((array_len,), dtype=dtype, buffer=mm,
    #                              offset=np.dtype(dtype).itemsize * start_idx)
    #             arr[:] = example['input_ids']
    #             mm.flush()
    #
    #     # Process each domain and each length range in one go
    #     for domain, length_groups in domain_length_example_dict.items():
    #         for len_range, tokenized_examples in length_groups.items():
    #             if len(tokenized_examples) > 0:
    #                 # Create a Dataset from the list of tokenized examples for the current length range
    #                 domain_dataset = HF_Dataset.from_dict({
    #                     'input_ids': [ex['input_ids'] for ex in tokenized_examples],
    #                     'len': [ex['len'] for ex in tokenized_examples]
    #                 })
    #
    #                 # Add 'len_offset' column for concatenation
    #                 domain_dataset = domain_dataset.add_column(
    #                     'len_offset', np.cumsum(domain_dataset['len'])
    #                 )
    #
    #                 # Save the tokenized examples for the current domain and length range
    #                 array_len = domain_dataset[-1]['len_offset']
    #                 filename = cache_dir / f'{domain}_{len_range}.bin'
    #                 subprocess.run(['truncate', '-s', str(array_len * np.dtype(dtype).itemsize),
    #                                 str(filename)], check=True)
    #
    #                 domain_dataset.map(
    #                     write_ids_to_disk,
    #                     fn_kwargs={'filename': filename},
    #                     batched=False,
    #                     num_proc=max(self.num_workers, 1),
    #                     desc=f"Concatenating {len_range} examples for {domain}",
    #                 )
    #
    #                 # Store the concatenated input_ids in memory map
    #                 concat_ids[f"{domain}_{len_range}"] = np.memmap(filename, dtype=dtype, mode='r', shape=(array_len,))
    #
    #     if cache_dir is not None:
    #         self._save_to_cache(concat_ids, tokenizer, cache_dir)
    #         if not self.use_shmem:
    #             for domain_group in concat_ids:
    #                 Path(cache_dir / f'{domain_group}.bin').unlink()
    #
    #     return concat_ids, tokenizer

    # def process_dataset(self):
    #     cache_dir = None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
    #     if cache_dir is not None and cache_dir.is_dir():
    #         return self._load_from_cache(cache_dir)
    #
    #     print('Start Loading!')
    #     raw_datasets = load_dataset(self.dataset_name, self.dataset_config_name)
    #     print('Loading Done!')
    #
    #     tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
    #     column_names = raw_datasets["train"].column_names  # e.g., 'text', 'meta'
    #     text_column_name = "text"
    #     meta_column_name = "meta"
    #
    #     assert self.add_eos
    #     dtype = np.uint16 if tokenizer.vocab_size < 64 * 1024 else np.int32
    #
    #     def tokenize_and_group(examples):
    #         tokens = tokenizer(
    #             examples[text_column_name],
    #             add_special_tokens=False,
    #             truncation=False,
    #             padding=False
    #         )
    #         if self.add_eos:
    #             tokens['input_ids'] = [ids + [tokenizer.eos_token_id] for ids in tokens['input_ids']]
    #
    #         categorized = {
    #             'input_ids': [],
    #             'domain': [],
    #             'length_category': [],
    #             'len': []
    #         }
    #
    #         for input_ids, meta in zip(tokens['input_ids'], examples[meta_column_name]):
    #             domain = meta.get('redpajama_set_name', 'unknown_domain')
    #             length = len(input_ids)
    #
    #             # Define length-based category keys
    #             if length < 10 * 1e3:
    #                 len_category = 'tok_len_below_10K'
    #             elif length < 100 * 1e3:
    #                 len_category = 'tok_len_10K_100K'
    #             elif length < 200 * 1e3:
    #                 len_category = 'tok_len_100K_200K'
    #             elif length < 500 * 1e3:
    #                 len_category = 'tok_len_200K_500K'
    #             elif length < 1000 * 1e3:
    #                 len_category = 'tok_len_500K_1M'
    #             else:
    #                 len_category = 'tok_len_above_1M'
    #
    #             categorized['input_ids'].append(input_ids)
    #             categorized['domain'].append(domain)
    #             categorized['length_category'].append(len_category)
    #             categorized['len'].append(length)
    #
    #         return categorized
    #     tokenized_and_categorized = raw_datasets['train'].map(
    #         tokenize_and_group,
    #         batched=True,
    #         num_proc=self.num_workers,
    #         remove_columns=column_names,
    #         desc="Tokenizing and grouping by domain and length",
    #     )
    #
    #     concat_ids = {}
    #     assert cache_dir is not None
    #     cache_dir.mkdir(parents=True, exist_ok=True)
    #
    #     # Define domains and length categories
    #     domains = [
    #         "RedPajamaCommonCrawl", "RedPajamaC4", "RedPajamaGithub",
    #         "RedPajamaBook", "RedPajamaArXiv", "RedPajamaWikipedia",
    #         "RedPajamaStackExchange"
    #     ]
    #     length_categories = [
    #         'tok_len_below_10K', 'tok_len_10K_100K', 'tok_len_100K_200K',
    #         'tok_len_200K_500K', 'tok_len_500K_1M', 'tok_len_above_1M'
    #     ]
    #
    #     def write_ids_to_disk(example, filename):
    #         with open(filename, 'r+b') as f:
    #             mm = mmap.mmap(f.fileno(), 0)
    #             start_idx = example['len_offset'] - len(example['input_ids'])
    #             array_len = len(example['input_ids'])
    #             arr = np.ndarray((array_len,), dtype=dtype, buffer=mm,
    #                              offset=np.dtype(dtype).itemsize * start_idx)
    #             arr[:] = example['input_ids']
    #             mm.flush()
    #
    #     def process_group(group_name, dataset):
    #         dataset = dataset.add_column(
    #             'len_offset', np.cumsum(dataset['len'])
    #         )
    #         array_len = dataset[-1]['len_offset']
    #
    #         filename = cache_dir / f'{group_name}.bin'
    #         subprocess.run(['truncate', '-s', str(array_len * np.dtype(dtype).itemsize), str(filename)], check=True)
    #
    #         dataset.map(
    #             write_ids_to_disk,
    #             fn_kwargs={'filename': filename},
    #             batched=False,
    #             num_proc=max(self.num_workers, 1),
    #             desc=f"Concatenating examples for {group_name}",
    #         )
    #         concat_ids[group_name] = np.memmap(filename, dtype=dtype, mode='r', shape=(array_len,))
    #
    #     # Separate and process by domain and length category
    #     for domain in domains:
    #         for len_category in length_categories:
    #             subset = tokenized_and_categorized.filter(
    #                 lambda x: x['domain'] == domain and x['length_category'] == len_category
    #             )
    #             if subset.num_rows > 0:
    #                 group_name = f"{domain}_{len_category}"
    #                 process_group(group_name, subset)
    #
    #     if cache_dir is not None:
    #         self._save_to_cache(concat_ids, tokenizer, cache_dir)
    #         if not self.use_shmem:
    #             for domain_group in concat_ids:
    #                 Path(cache_dir / f'{domain_group}.bin').unlink()
    #
    #     return concat_ids, tokenizer

    # def process_dataset(self):
    #     cache_dir = None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
    #     if cache_dir is not None and cache_dir.is_dir():
    #         return self._load_from_cache(cache_dir)
    #
    #     print('Start Loading!')
    #     raw_datasets = load_dataset(self.dataset_name, self.dataset_config_name)
    #     print('Loading Done!')
    #
    #     tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
    #     column_names = raw_datasets["train"].column_names  # 'text', 'meta'
    #     text_column_name = "text"
    #     meta_column_name = "meta"
    #     dtype = np.uint16 if tokenizer.vocab_size < 64 * 1024 else np.int32
    #
    #     def tokenize_and_group(examples):
    #         # Tokenize and add EOS if needed
    #         tokens = tokenizer(
    #             examples[text_column_name],
    #             add_special_tokens=False,
    #             truncation=False,
    #             padding=False
    #         )
    #         if self.add_eos:
    #             tokens['input_ids'] = [ids + [tokenizer.eos_token_id] for ids in tokens['input_ids']]
    #
    #         # Initialize lists to store flat structure
    #         domains, length_categories, input_ids_list, lengths = [], [], [], []
    #
    #         # Categorize each example by domain and length category
    #         for input_ids, meta in zip(tokens['input_ids'], examples[meta_column_name]):
    #             domain = meta.get('redpajama_set_name', 'unknown_domain')
    #             length = len(input_ids)
    #
    #             # Define length-based category keys
    #             if length < 10 * 1e3:
    #                 len_category = 'tok_len_below_10K'
    #             elif length < 100 * 1e3:
    #                 len_category = 'tok_len_10K_100K'
    #             elif length < 200 * 1e3:
    #                 len_category = 'tok_len_100K_200K'
    #             elif length < 500 * 1e3:
    #                 len_category = 'tok_len_200K_500K'
    #             elif length < 1000 * 1e3:
    #                 len_category = 'tok_len_500K_1M'
    #             else:
    #                 len_category = 'tok_len_above_1M'
    #
    #             # Append to lists to maintain a flat structure
    #             domains.append(domain)
    #             length_categories.append(len_category)
    #             input_ids_list.append(input_ids)
    #             lengths.append(length)
    #
    #         return {
    #             'domain': domains,
    #             'length_category': length_categories,
    #             'input_ids': input_ids_list,
    #             'len': lengths
    #         }
    #
    #     # Map the function to apply tokenization and categorization
    #     tokenized_and_categorized = raw_datasets['train'].map(
    #         tokenize_and_group,
    #         batched=True,
    #         num_proc=self.num_workers,
    #         remove_columns=column_names,
    #         desc="Tokenizing and grouping by domain and length",
    #     )
    #
    #     # # Group examples by `domain` and `length_category`
    #     grouped_results = defaultdict(list)
    #     for i in tqdm(range(len(tokenized_and_categorized)), desc="Filtering by domain and length"):
    #         key = f"{tokenized_and_categorized['domain'][i]}_{tokenized_and_categorized['length_category'][i]}"
    #         grouped_results[key].append({
    #             'input_ids': tokenized_and_categorized['input_ids'][i],
    #             'len': tokenized_and_categorized['len'][i]
    #         })
    #
    #     # Prepare for saving concatenated examples to disk
    #     concat_ids = {}
    #     assert cache_dir is not None
    #     cache_dir.mkdir(parents=True, exist_ok=True)
    #
    #     def write_ids_to_disk(example, filename):
    #         with open(filename, 'r+b') as f:
    #             mm = mmap.mmap(f.fileno(), 0)
    #             start_idx = example['len_offset'] - len(example['input_ids'])
    #             array_len = len(example['input_ids'])
    #             arr = np.ndarray((array_len,), dtype=dtype, buffer=mm,
    #                              offset=np.dtype(dtype).itemsize * start_idx)
    #             arr[:] = example['input_ids']
    #             mm.flush()
    #
    #     # Process each domain-length group
    #     for key, examples in grouped_results.items():
    #         # Create a Dataset from the examples
    #         dataset = HF_Dataset.from_dict(
    #             {k: [example[k] for example in examples] for k in examples[0].keys()}
    #         )
    #
    #         # Add cumulative lengths for len_offset
    #         dataset = dataset.add_column('len_offset', np.cumsum(dataset['len']))
    #         array_len = dataset[-1]['len_offset']
    #
    #         # Save the tokenized examples for this group
    #         filename = cache_dir / f'{key}.bin'
    #         subprocess.run(['truncate', '-s', str(array_len * np.dtype(dtype).itemsize), str(filename)], check=True)
    #
    #         # Write tokenized examples to disk
    #         dataset.map(
    #             write_ids_to_disk,
    #             fn_kwargs={'filename': filename},
    #             batched=False,
    #             num_proc=max(self.num_workers, 1),
    #             desc=f"Concatenating {key} examples",
    #         )
    #         concat_ids[key] = np.memmap(filename, dtype=dtype, mode='r', shape=(array_len,))
    #
    #     if cache_dir is not None:
    #         self._save_to_cache(concat_ids, tokenizer, cache_dir)
    #         if not self.use_shmem:
    #             for domain_group in concat_ids:
    #                 Path(cache_dir / f'{domain_group}.bin').unlink()
    #
    #     return concat_ids, tokenizer

    def process_dataset(self):
        cache_dir = None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        if cache_dir is not None and cache_dir.is_dir():
            return self._load_from_cache(cache_dir)

        print('Start Loading!')
        raw_datasets = load_dataset(self.dataset_name)
        print('Loading Done!')

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        column_names = raw_datasets["train"].column_names  # 'text', 'meta'
        text_column_name = "text"
        meta_column_name = "meta"
        dtype = np.uint16 if tokenizer.vocab_size < 64 * 1024 else np.int32

        def tokenize_and_group(examples):
            tokens = tokenizer(examples[text_column_name])  # llama3: will add bos, but not eos

            if self.pad_to_multiple_of > 0:
                def pad_to_multiple(tokens, pad_token_id, multiple):
                    length = len(tokens)
                    padding_length = (multiple - length % multiple) % multiple
                    return tokens + [pad_token_id] * padding_length

                def tokenize_pad_to_multiple(input_ids):
                    if self.add_eos:
                        input_ids = input_ids + [tokenizer.eos_token_id]
                    padded_input_ids = pad_to_multiple(input_ids, tokenizer.bos_token_id, self.pad_to_multiple_of)
                    return padded_input_ids

                tokens['input_ids'] = [tokenize_pad_to_multiple(ids) for ids in tokens['input_ids']]

            else:
                if self.add_eos:
                    tokens['input_ids'] = [ids + [tokenizer.eos_token_id] for ids in tokens['input_ids']]

            # Initialize lists to store flat structure
            domains, length_categories, input_ids_list, lengths = [], [], [], []

            # Categorize each example by domain and length category
            for input_ids, meta in zip(tokens['input_ids'], examples[meta_column_name]):
                domain = meta.get('redpajama_set_name', 'unknown_domain')
                length = len(input_ids)

                # Define length-based category keys
                if length < 10 * 1e3:
                    len_category = 'tok_len_below_10K'
                elif length < 100 * 1e3:
                    len_category = 'tok_len_10K_100K'
                elif length < 200 * 1e3:
                    len_category = 'tok_len_100K_200K'
                elif length < 500 * 1e3:
                    len_category = 'tok_len_200K_500K'
                elif length < 1000 * 1e3:
                    len_category = 'tok_len_500K_1M'
                else:
                    len_category = 'tok_len_above_1M'

                # Append to lists to maintain a flat structure
                domains.append(domain)
                length_categories.append(len_category)
                input_ids_list.append(input_ids)
                lengths.append(length)

            return {
                'domain': domains,
                'length_category': length_categories,
                'input_ids': input_ids_list,
                'len': lengths
            }

        # Map the function to apply tokenization and categorization
        tokenized_and_categorized = raw_datasets['train'].map(
            tokenize_and_group,
            batched=True,
            num_proc=self.num_workers,
            remove_columns=column_names,
            desc="Tokenizing and grouping by domain and length",
        )

        print('Started grouping by Pandas')
        st = time.time()
        # Convert the tokenized results to a pandas DataFrame for efficient processing
        df = pd.DataFrame(tokenized_and_categorized)
        # Group examples by `domain` and `length_category`
        grouped_results = defaultdict(list)
        for (domain, length_category), group in df.groupby(['domain', 'length_category']):
            grouped_results[f"{domain}_{length_category}"] = group[['input_ids', 'len']].to_dict('records')
        grouping_time = time.time() - st
        print(f'Grouping done. Time: {grouping_time / 60:.1f} min')

        # Prepare for saving concatenated examples to disk
        concat_ids = {}
        assert cache_dir is not None
        cache_dir.mkdir(parents=True, exist_ok=True)

        def write_ids_to_disk(example, filename):
            with open(filename, 'r+b') as f:
                mm = mmap.mmap(f.fileno(), 0)
                start_idx = example['len_offset'] - len(example['input_ids'])
                array_len = len(example['input_ids'])
                arr = np.ndarray((array_len,), dtype=dtype, buffer=mm,
                                 offset=np.dtype(dtype).itemsize * start_idx)
                arr[:] = example['input_ids']
                mm.flush()

        # Process each domain-length group
        for key, examples in grouped_results.items():
            # Create a Dataset from the examples
            dataset = HF_Dataset.from_dict(
                {k: [example[k] for example in examples] for k in examples[0].keys()}
            )

            # Add cumulative lengths for len_offset
            dataset = dataset.add_column('len_offset', np.cumsum(dataset['len']))
            array_len = dataset[-1]['len_offset']

            # Save the tokenized examples for this group
            filename = cache_dir / f'{key}.bin'
            subprocess.run(['truncate', '-s', str(array_len * np.dtype(dtype).itemsize), str(filename)], check=True)

            # Write tokenized examples to disk
            dataset.map(
                write_ids_to_disk,
                fn_kwargs={'filename': filename},
                batched=False,
                num_proc=max(self.num_workers, 1),
                desc=f"Concatenating {key} examples",
            )
            concat_ids[key] = np.memmap(filename, dtype=dtype, mode='r', shape=(array_len,))

        if cache_dir is not None:
            self._save_to_cache(concat_ids, tokenizer, cache_dir)
            if not self.use_shmem:
                for domain_group in concat_ids:
                    Path(cache_dir / f'{domain_group}.bin').unlink()

        return concat_ids, tokenizer


    def _save_to_cache(self, concat_ids, tokenizer, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        # logger.info(f'Saving to cache at {str(cache_dir)}')
        print(f'Saving to cache at {str(cache_dir)}')
        for k, v in concat_ids.items():
            np.save(cache_dir / f'{k}.npy', v)
        with open(cache_dir / 'tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        # logger.info(f'Load from cache at {str(cache_dir)}')
        print(f'Load from cache at {str(cache_dir)}')
        if self.dataset_name == 'books3_splitted_finetune':
            concat_ids = {split: np.load(cache_dir / f'{split}.npy', mmap_mode='r')
                          for split in ['train', 'finetune', 'validation', 'test']}
        else:
            concat_ids = {split: np.load(cache_dir / f'{split}.npy', mmap_mode='r')
                          for split in ['train', 'validation', 'test']}
        with open(cache_dir / 'tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return concat_ids, tokenizer

    @property
    def _cache_dir_name(self):
        return f'tokenizer_name-{self.tokenizer_name}-val_ratio-{self.val_ratio}-val_split_seed-{self.val_split_seed}-add_eos-{self.add_eos}-detokenize-{self.detokenize}'

    def train_dataloader(self, mode='pretrain', *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        if mode == 'pretrain':
            selected_dataset = self.dataset_train
        elif mode == 'finetune':
            selected_dataset = self.dataset_finetune
        else:
            raise NotImplementedError(f'Mode {mode} not implemented!')

        if self.shuffle and self.fault_tolerant:
            shuffle = False
            sampler = (FaultTolerantDistributedSampler(selected_dataset) if self.ddp
                       else RandomFaultTolerantSampler(selected_dataset))
            # TD [2022-08-06]: Only the DDP sampler supports fast-forwarding for now
            # We assume that it's being resumed with the same number of GPUs
            if self.ddp and self.fast_forward_epochs is not None and self.fast_forward_batches is not None:
                sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter': self.fast_forward_batches * self.batch_size
                })
        else:
            shuffle = self.shuffle
            sampler = None
        return self._data_loader(selected_dataset, batch_size=self.batch_size,
                                 shuffle=shuffle, sampler=sampler)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        return self._data_loader(self.dataset_test, batch_size=self.batch_size_eval)

    def _data_loader(self, dataset: Dataset, batch_size: int, shuffle: bool = False,
                     sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=1,  # Data is already in memory, we don't need many workers
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            # persistent_workers=True
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
            # TD [2022-08-07] ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
            # behind, so we're using the optimizer's progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']
        # At this point the train loader hasn't been constructed yet
