import pdb

# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
from itertools import chain
from pathlib import Path
import pickle
from typing import Any, List, Union
import subprocess
import mmap
from collections import defaultdict

from multiprocessing.shared_memory import SharedMemory

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


# https://github.com/numpy/numpy/issues/18294
class SHMArray(np.ndarray): #copied from https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array

    def __new__(cls, input_array, shm=None):
        obj = np.asarray(input_array).view(cls)
        obj.shm = shm
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.shm = getattr(obj, 'shm', None)


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
    #
    #     assert self.add_eos
    #     add_eos = lambda seq: (seq + tokenizer.eos_token) if seq else seq
    #     add_eos_batched = lambda seqs: [add_eos(seq) for seq in seqs]
    #     tokenize = lambda example: tokenizer(add_eos_batched(example[text_column_name]))
    #
    #     dtype = np.uint16 if tokenizer.vocab_size < 64 * 1024 else np.int32
    #
    #     def tokenize_concat(examples):
    #         input_ids = np.fromiter(chain(*tokenize(examples)['input_ids']), dtype=dtype)
    #         # Need to return a list since we're doing batched processing
    #         return {'input_ids': [input_ids], 'len': [len(input_ids)]}
    #
    #     tokenized_datasets = raw_datasets.map(
    #         tokenize_concat,
    #         batched=True,
    #         num_proc=max(self.num_workers, 1),
    #         remove_columns=column_names,
    #         desc="Running tokenizer on dataset",
    #     )
    #
    #     # Use disk
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
    #     for name, ds in tokenized_datasets.items():
    #         tokenized_datasets[name] = ds.add_column('len_offset', np.cumsum(ds['len']))
    #         array_len = tokenized_datasets[name][-1]['len_offset']
    #
    #         filename = cache_dir / f'{name}.bin'
    #         subprocess.run(['truncate', '-s', str(array_len * np.dtype(dtype).itemsize),
    #                         str(filename)], check=True)
    #
    #         tokenized_datasets[name].map(
    #             write_ids_to_disk,
    #             fn_kwargs={'filename': filename},
    #             batched=False,
    #             num_proc=max(self.num_workers, 1),
    #             desc="Concatenating examples",
    #         )
    #         concat_ids[name] = np.memmap(filename, dtype=dtype, mode='r', shape=(array_len,))
    #
    #     if cache_dir is not None:
    #         self._save_to_cache(concat_ids, tokenizer, cache_dir)
    #         if not self.use_shmem:
    #             for name in concat_ids:
    #                 Path(cache_dir / f'{name}.bin').unlink()
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
    #     add_eos = lambda seq: (seq + tokenizer.eos_token) if seq else seq
    #     add_eos_batched = lambda seqs: [add_eos(seq) for seq in seqs]
    #     tokenize = lambda example: tokenizer(add_eos_batched(example[text_column_name]))
    #
    #     dtype = np.uint16 if tokenizer.vocab_size < 64 * 1024 else np.int32
    #
        # def tokenize_concat(examples):
        #     input_ids = np.fromiter(chain(*tokenize(examples)['input_ids']), dtype=dtype)
        #     # Return a list for batched processing
        #     return {'input_ids': [input_ids], 'len': [len(input_ids)]}
    #
    #     # Process each domain separately
    #     domains = [
    #         "RedPajamaCommonCrawl", "RedPajamaC4", "RedPajamaGithub",
    #         "RedPajamaBook", "RedPajamaArXiv", "RedPajamaWikipedia",
    #         "RedPajamaStackExchange"
    #     ]
    #
    #     concat_ids = {}
    #     assert cache_dir is not None
    #     cache_dir.mkdir(parents=True, exist_ok=True)
    #
    #     for domain in domains:
    #         # Filter dataset for the current domain
    #         domain_datasets = raw_datasets.filter(lambda example: example[meta_column_name]['redpajama_set_name'] == domain)
    #
    #         tokenized_datasets = domain_datasets.map(
    #             tokenize_concat,
    #             batched=True,
    #             num_proc=max(self.num_workers, 1),
    #             remove_columns=column_names,
    #             desc=f"Running tokenizer on {domain} dataset",
    #         )
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
    #         # Add 'len_offset' to each split in tokenized_datasets
    #         for split in tokenized_datasets.keys():
    #             tokenized_datasets[split] = tokenized_datasets[split].add_column(
    #                 'len_offset', np.cumsum(tokenized_datasets[split]['len'])
    #             )
    #             array_len = tokenized_datasets[split][-1]['len_offset']
    #
    #             # Save the tokenized examples for this domain and split
    #             filename = cache_dir / f'{domain}_{split}.bin'
    #             subprocess.run(['truncate', '-s', str(array_len * np.dtype(dtype).itemsize),
    #                             str(filename)], check=True)
    #
    #             tokenized_datasets[split].map(
    #                 write_ids_to_disk,
    #                 fn_kwargs={'filename': filename},
    #                 batched=False,
    #                 num_proc=max(self.num_workers, 1),
    #                 desc=f"Concatenating examples for {domain} {split}",
    #             )
    #             concat_ids[f"{domain}_{split}"] = np.memmap(filename, dtype=dtype, mode='r', shape=(array_len,))
    #
    #
    #     if cache_dir is not None:
    #         self._save_to_cache(concat_ids, tokenizer, cache_dir)
    #         for domain in concat_ids:
    #             Path(cache_dir / f'{domain}.bin').unlink()
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
    #     add_eos = lambda seq: (seq + tokenizer.eos_token) if seq else seq
    #     add_eos_batched = lambda seqs: [add_eos(seq) for seq in seqs]
    #     tokenize = lambda example: tokenizer(add_eos_batched(example[text_column_name]))
    #
    #     dtype = np.uint16 if tokenizer.vocab_size < 64 * 1024 else np.int32
    #
    #     def tokenize_concat(examples):
    #         input_ids = np.fromiter(chain(*tokenize(examples)['input_ids']), dtype=dtype)
    #         # Return a list for batched processing
    #         return {'input_ids': [input_ids], 'len': [len(input_ids)]}
    #
    #     # Initialize a defaultdict to store grouped data by domain
    #     domain_datasets = defaultdict(list)
    #
    #     # Group examples by their 'meta' field in a single pass
    #     # TODO: not handle val and test for now
    #     for example in raw_datasets['train']:
    #         domain = example[meta_column_name]['redpajama_set_name']
    #         domain_datasets[domain].append(example)
    #
    #     concat_ids = {}
    #     assert cache_dir is not None
    #     cache_dir.mkdir(parents=True, exist_ok=True)
    #
    #     # Now process each domain's dataset
    #     for domain, examples in domain_datasets.items():
    #         # Create a Dataset from the list of examples
    #         domain_dataset = HF_Dataset.from_dict(
    #             {key: [example[key] for example in examples] for key in examples[0].keys()})
    #
    #         tokenized_datasets = domain_dataset.map(
    #             tokenize_concat,
    #             batched=True,
    #             num_proc=max(self.num_workers, 1),
    #             remove_columns=column_names,
    #             desc=f"Running tokenizer on {domain} dataset",
    #         )
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
    #         # Add 'len_offset' to the tokenized dataset
    #         tokenized_datasets = tokenized_datasets.add_column(
    #             'len_offset', np.cumsum(tokenized_datasets['len'])
    #         )
    #         array_len = tokenized_datasets[-1]['len_offset']
    #
    #         # Save the tokenized examples for this domain
    #         filename = cache_dir / f'{domain}.bin'
    #         subprocess.run(['truncate', '-s', str(array_len * np.dtype(dtype).itemsize),
    #                         str(filename)], check=True)
    #
    #         tokenized_datasets.map(
    #             write_ids_to_disk,
    #             fn_kwargs={'filename': filename},
    #             batched=False,
    #             num_proc=max(self.num_workers, 1),
    #             desc=f"Concatenating examples for {domain}",
    #         )
    #         concat_ids[domain] = np.memmap(filename, dtype=dtype, mode='r', shape=(array_len,))
    #
    #     if cache_dir is not None:
    #         self._save_to_cache(concat_ids, tokenizer, cache_dir)
    #         if not self.use_shmem:
    #             for domain in concat_ids:
    #                 Path(cache_dir / f'{domain}.bin').unlink()
    #
    #     return concat_ids, tokenizer

    def process_dataset(self):
        cache_dir = None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        print('Start Loading!')
        raw_datasets = load_dataset(self.dataset_name, self.dataset_config_name)
        print('Loading Done!')

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)

        column_names = raw_datasets["train"].column_names  # 'text', 'meta'
        text_column_name = "text"
        meta_column_name = "meta"

        assert self.add_eos
        # add_eos = lambda seq: (seq + tokenizer.eos_token) if seq else seq
        # add_eos_batched = lambda seqs: [add_eos(seq) for seq in seqs]
        # tokenize = lambda example: tokenizer(add_eos_batched(example[text_column_name]))

        dtype = np.uint16 if tokenizer.vocab_size < 64 * 1024 else np.int32

        def tokenize_concat(examples):
            input_ids = [tokenizer(seq + tokenizer.eos_token)['input_ids'] for seq in examples[text_column_name]]
            lengths = [len(ids) for ids in input_ids]
            return {'input_ids': input_ids, 'len': lengths}

        # Initialize a defaultdict to store grouped data by domain
        domain_datasets = defaultdict(list)

        # Group examples by their 'meta' field in a single pass
        for example in raw_datasets['train']:
            domain = example[meta_column_name]['redpajama_set_name']
            domain_datasets[domain].append(example)

        concat_ids = {}
        assert cache_dir is not None
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Process each domain's dataset
        for domain, examples in domain_datasets.items():
            # Create a Dataset from the list of examples
            domain_dataset = HF_Dataset.from_dict(
                {key: [example[key] for example in examples] for key in examples[0].keys()})

            # Tokenize the examples and separate them by length (<1000 tokens vs >=1000 tokens)
            tokenized_datasets = domain_dataset.map(
                tokenize_concat,
                batched=True,
                num_proc=max(self.num_workers, 1),
                remove_columns=column_names,
                desc=f"Running tokenizer on {domain} dataset",
            )

            # Separate examples into two groups based on their length
            short_examples = tokenized_datasets.filter(lambda x: x['len'] < 1000)
            long_examples = tokenized_datasets.filter(lambda x: x['len'] >= 1000)

            def write_ids_to_disk(example, filename):
                with open(filename, 'r+b') as f:
                    mm = mmap.mmap(f.fileno(), 0)
                    start_idx = example['len_offset'] - len(example['input_ids'])
                    array_len = len(example['input_ids'])
                    arr = np.ndarray((array_len,), dtype=dtype, buffer=mm,
                                     offset=np.dtype(dtype).itemsize * start_idx)
                    arr[:] = example['input_ids']
                    mm.flush()

            # Function to process a group (short or long) of examples
            def process_group(group_name, dataset):
                dataset = dataset.add_column(
                    'len_offset', np.cumsum(dataset['len'])
                )
                array_len = dataset[-1]['len_offset']

                # Save the tokenized examples for this group (short/long) and domain
                filename = cache_dir / f'{domain}_{group_name}.bin'
                subprocess.run(['truncate', '-s', str(array_len * np.dtype(dtype).itemsize),
                                str(filename)], check=True)

                dataset.map(
                    write_ids_to_disk,
                    fn_kwargs={'filename': filename},
                    batched=False,
                    num_proc=max(self.num_workers, 1),
                    desc=f"Concatenating {group_name} examples for {domain}",
                )
                concat_ids[f"{domain}_{group_name}"] = np.memmap(filename, dtype=dtype, mode='r', shape=(array_len,))

            # Process short examples
            if len(short_examples['len']) > 0:
                process_group('short', short_examples)  # TODO: handle num_rows=0

            # Process long examples
            if len(long_examples['len']) > 0:
                process_group('long', long_examples)

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
