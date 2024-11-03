"""
@Xinhao
c3d-standard-180 180 vCPUs 720GB mem --> num_workers=16
"""
import os
from pathlib import Path
current_dir = Path(__file__).parent.absolute()


import pytest
import shutil
import subprocess

import torch
import numpy as np

import dotenv

from src.datamodules.language_modeling_hf import LMDataModule

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


def div_up(x: int, y: int) -> int:
    return (x + y - 1) // y


# https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python/55423170#55423170
def num_cpu_cores():
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except ImportError:
        return len(os.sched_getaffinity(0))


class TestLMDataModule:

    def test_wikitext2(self):
        batch_size = 7
        dataset_name = 'wikitext'
        dataset_config_name = 'wikitext-2-raw-v1'
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data'))
        cache_dir = data_dir / 'wikitext-2' / 'cache'
        max_length = 1024
        datamodule = LMDataModule(dataset_name, tokenizer_name='gpt2',
                                  dataset_config_name=dataset_config_name,
                                  max_length=max_length, cache_dir=cache_dir,
                                  add_eos=False, batch_size=batch_size, num_workers=4)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        train_len = 2391884
        val_len = 247289
        test_len = 283287
        assert len(train_loader) == div_up((train_len - 1) // max_length, batch_size)
        assert len(val_loader) == div_up((val_len - 1) // max_length, batch_size)
        assert len(test_loader) == div_up((test_len - 1) // max_length, batch_size)
        for loader in [train_loader, val_loader, test_loader]:
            x, y = next(iter(loader))
            assert x.dim() == 2
            assert x.shape == (batch_size, max_length)
            assert x.dtype == torch.long
            assert torch.allclose(x[:, 1:], y[:, :-1])

    def test_wikitext103(self):
        batch_size = 7
        dataset_name = 'wikitext'
        dataset_config_name = 'wikitext-103-raw-v1'
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data'))
        cache_dir = data_dir / 'wikitext-103' / 'cache'
        max_length = 1024
        datamodule = LMDataModule(dataset_name, tokenizer_name='gpt2',
                                  dataset_config_name=dataset_config_name,
                                  max_length=max_length, cache_dir=cache_dir,
                                  add_eos=False, batch_size=batch_size, num_workers=4)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        train_len = 117920140
        val_len = 247289
        test_len = 283287
        assert len(train_loader) == div_up((train_len - 1) // max_length, batch_size)
        assert len(val_loader) == div_up((val_len - 1) // max_length, batch_size)
        assert len(test_loader) == div_up((test_len - 1) // max_length, batch_size)
        for loader in [train_loader, val_loader, test_loader]:
            x, y = next(iter(loader))
            assert x.dim() == 2
            assert x.shape == (batch_size, max_length)
            assert x.dtype == torch.long
            assert torch.allclose(x[:, 1:], y[:, :-1])

    def test_openwebtext(self):
        batch_size = 8
        dataset_name = 'openwebtext'
        dataset_config_name = None
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data'))
        cache_dir = data_dir / 'openwebtext' / 'cache'
        max_length = 1024
        datamodule = LMDataModule(dataset_name, tokenizer_name='gpt2',
                                  dataset_config_name=dataset_config_name,
                                  max_length=max_length, cache_dir=cache_dir,
                                  add_eos=True, batch_size=batch_size,
                                  num_workers=num_cpu_cores() // 2)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        train_len = 9035582198
        val_len = 4434897
        test_len = 4434897
        assert len(train_loader) == div_up((train_len - 1) // max_length, batch_size)
        assert len(val_loader) == div_up((val_len - 1) // max_length, batch_size)
        assert len(test_loader) == div_up((test_len - 1) // max_length, batch_size)
        for loader in [train_loader, val_loader, test_loader]:
            x, y = next(iter(loader))
            assert x.dim() == 2
            assert x.shape == (batch_size, max_length)
            assert x.dtype == torch.long
            assert torch.allclose(x[:, 1:], y[:, :-1])

    def test_lambada(self):
        batch_size = 8
        dataset_name = 'lambada'
        dataset_config_name = None
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data'))
        cache_dir = data_dir / 'lambada' / 'cache'
        max_length = 1024
        datamodule = LMDataModule(dataset_name, tokenizer_name='gpt2',
                                  dataset_config_name=dataset_config_name,
                                  max_length=max_length, cache_dir=cache_dir,
                                  add_eos=True, batch_size=batch_size,
                                  num_workers=64)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        train_len = 9035582198
        val_len = 4434897
        test_len = 4434897
        assert len(train_loader) == div_up((train_len - 1) // max_length, batch_size)
        assert len(val_loader) == div_up((val_len - 1) // max_length, batch_size)
        assert len(test_loader) == div_up((test_len - 1) // max_length, batch_size)
        for loader in [train_loader, val_loader, test_loader]:
            x, y = next(iter(loader))
            assert x.dim() == 2
            assert x.shape == (batch_size, max_length)
            assert x.dtype == torch.long
            assert torch.allclose(x[:, 1:], y[:, :-1])

    def test_the_pile(self):
        batch_size = 8
        dataset_name = 'the_pile'
        dataset_config_name = None
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data'))
        cache_dir = data_dir / 'the_pile' / 'cache'
        max_length = 2048
        # Dataset is too large to fit into memory, need to use disk for concatenation
        datamodule = LMDataModule(dataset_name, tokenizer_name='gpt2',
                                  dataset_config_name=dataset_config_name,
                                  max_length=max_length, cache_dir=cache_dir,
                                  add_eos=True, batch_size=batch_size,
                                  num_workers=num_cpu_cores() // 2, use_shmem=False)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        train_len = 374337375694
        val_len = 383326395
        test_len = 373297018
        assert len(train_loader) == div_up((train_len - 1) // max_length, batch_size)
        assert len(val_loader) == div_up((val_len - 1) // max_length, batch_size)
        assert len(test_loader) == div_up((test_len - 1) // max_length, batch_size)
        for loader in [train_loader, val_loader, test_loader]:
            x, y = next(iter(loader))
            assert x.dim() == 2
            assert x.shape == (batch_size, max_length)
            assert x.dtype == torch.long
            assert torch.allclose(x[:, 1:], y[:, :-1])

    def test_pg19(self):
        batch_size = 8
        dataset_name = 'pg19'
        dataset_config_name = None
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data'))
        cache_dir = data_dir / 'pg19' / 'cache'
        max_length = 2048
        # Dataset is too large to fit into memory, need to use disk for concatenation
        datamodule = LMDataModule(dataset_name, tokenizer_name='gpt2',
                                  dataset_config_name=dataset_config_name,
                                  max_length=max_length, cache_dir=cache_dir,
                                  add_eos=True, batch_size=batch_size,
                                  num_workers=num_cpu_cores() // 2)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        train_len = 3066544128
        val_len = 4653056
        test_len = 10584064
        assert len(train_loader) == div_up((train_len - 1) // max_length, batch_size)
        assert len(val_loader) == div_up((val_len - 1) // max_length, batch_size)
        assert len(test_loader) == div_up((test_len - 1) // max_length, batch_size)
        for loader in [train_loader, val_loader, test_loader]:
            x, y = next(iter(loader))
            assert x.dim() == 2
            assert x.shape == (batch_size, max_length)
            assert x.dtype == torch.long
            assert torch.allclose(x[:, 1:], y[:, :-1])

    def test_books(self, dataset_name):
        # batch_size = 8
        # dataset_name = 'books3_splitted_finetune'  # useless
        # dataset_config_name = None
        # cache_dir = Path('/mnt/disks/persistent/books3_splitted_finetune')  # path to save tokenized dataset
        # raw_json_path = '/mnt/disks/persistent/lwm_raw/lwm_text_data/combined_books.jsonl'
        # finetune_ratio = 0.167  # 1/6 of full train set becomes finetune set, 5/6 is pre-train set
        # max_length = 2048  # only useful for deciding chunking data for sampler idx, won't affect tokenization
        # # num_workers = num_cpu_cores() // 2
        # num_workers = 1
        # # Dataset is too large to fit into memory, need to use disk for concatenation
        # datamodule = LMDataModule(dataset_name, tokenizer_name='meta-llama/Llama-2-7b-hf',
        #                           dataset_config_name=dataset_config_name,
        #                           max_length=max_length, cache_dir=cache_dir,
        #                           add_eos=True, batch_size=batch_size,
        #                           num_workers=num_workers, use_shmem=False,
        #                           raw_json_path=raw_json_path, finetune_ratio=finetune_ratio)
        # datamodule.prepare_data()
        # datamodule.setup(stage='fit')
        # train_loader = datamodule.train_dataloader()
        # finetune_loader = datamodule.train_dataloader(mode='finetune')
        # val_loader = datamodule.val_dataloader()
        # print('ctx=2k Train loader length: ', len(train_loader))
        # print('ctx=2k Finetune loader length: ', len(finetune_loader))
        # print('ctx=2k Val loader length: ', len(val_loader))

        # batch_size = 8
        # dataset_name = 'books3_pad_bos_16'  # useless
        # dataset_config_name = None
        # cache_dir = Path('/mnt/disks/persistent/books3_pad_bos_16')  # path to save tokenized dataset
        # raw_json_path = '/mnt/disks/persistent/lwm_raw/lwm_text_data/combined_books.jsonl'
        # max_length = 2048  # only useful for deciding chunking data for sampler idx, won't affect tokenization
        # # num_workers = num_cpu_cores() // 2
        # num_workers = 1
        # chunk_size = 16
        # # Dataset is too large to fit into memory, need to use disk for concatenation
        # datamodule = LMDataModule(dataset_name, tokenizer_name='meta-llama/Llama-2-7b-hf',
        #                           dataset_config_name=dataset_config_name,
        #                           max_length=max_length, cache_dir=cache_dir,
        #                           add_eos=True, batch_size=batch_size,
        #                           num_workers=num_workers, use_shmem=False,
        #                           raw_json_path=raw_json_path, pad_to_multiple_of=chunk_size)
        # datamodule.prepare_data()
        # datamodule.setup(stage='fit')
        # train_loader = datamodule.train_dataloader()
        # val_loader = datamodule.val_dataloader()
        # print('ctx=2k Train loader length: ', len(train_loader))
        # print('ctx=2k Val loader length: ', len(val_loader))

        # from transformers import (
        #     AutoTokenizer,
        #     BitsAndBytesConfig,
        #     LlamaForCausalLM,
        #     LlamaConfig,
        # )
        # dataset_name_list = ['books3_10k_100k', 'books3_100k_200k',
        #                      'books3_200k_500k', 'books3_500k_1M',
                             # 'books3_1M_plus'
                             # ]
        # dataset_name_list = ['books3_1M_plus']
        dataset_name_list = [dataset_name]
        for dataset_name in dataset_name_list:
            print(dataset_name)
            batch_size = 8
            dataset_config_name = None
            cache_dir = Path(f'/persistent/datasets/{dataset_name}_pad_bos_16')  # path to save tokenized dataset
            raw_json_path = f'/books3_jsonl/{dataset_name}.jsonl'
            max_length = 2048  # only useful for deciding chunking data for sampler idx, won't affect tokenization
            num_workers = 16
            chunk_size = 16
            # Dataset is too large to fit into memory, need to use disk for concatenation
            datamodule = LMDataModule(dataset_name, tokenizer_name='meta-llama/Meta-Llama-3.1-8B',
                                      dataset_config_name=dataset_config_name,
                                      max_length=max_length, cache_dir=cache_dir,
                                      add_eos=True, batch_size=batch_size,
                                      num_workers=num_workers, use_shmem=False,
                                      raw_json_path=raw_json_path, pad_to_multiple_of=chunk_size)
            datamodule.prepare_data()
            datamodule.setup(stage='fit')
            train_loader = datamodule.train_dataloader()
            val_loader = datamodule.val_dataloader()
            print('ctx=2k Train loader length: ', len(train_loader))
            print('ctx=2k Val loader length: ', len(val_loader))

    # def test_slimpajama(self):
    #     from src.datamodules.language_modeling_SPJ_hf import LMDataModule
    #     dataset_name = '/juice5/scr5/nlp/mttt/datasets/SlimPajama-627B-test'
    #     dataset_config_name = None
    #     cache_dir = Path(f'/juice5/scr5/nlp/mttt/datasets/SlimPajama-627B-llama3-tokenized')  # path to save tokenized dataset
    #     batch_size = 8
    #     max_length = 2048
    #     num_workers = num_cpu_cores() // 2
    #     chunk_size = 16
    #     # Dataset is too large to fit into memory, need to use disk for concatenation
    #     datamodule = LMDataModule(dataset_name, tokenizer_name='meta-llama/Meta-Llama-3.1-8B',
    #                               dataset_config_name=dataset_config_name,
    #                               max_length=max_length, cache_dir=cache_dir,
    #                               add_eos=True, batch_size=batch_size,
    #                               num_workers=num_workers, use_shmem=False,
    #                               raw_json_path=None, pad_to_multiple_of=chunk_size)
    #     datamodule.prepare_data()

    def test_slimpajama(self, chunk_name):
        from src.datamodules.language_modeling_SPJ_hf import LMDataModule
        
        data_path_prefix = '/juice5/scr5/nlp/mttt/datasets'
        # data_path_prefix = '/persistent_1/datasets'

        dataset_name = os.path.join(data_path_prefix, f'SlimPajama-627B-tmp/{chunk_name}')
        os.makedirs(dataset_name, exist_ok=True)
        subprocess.run(['cp', '-r', os.path.join(data_path_prefix, f'SlimPajama-627B-tmp/test'), dataset_name], check=True)
        subprocess.run(['cp', '-r', os.path.join(data_path_prefix, f'SlimPajama-627B-tmp/validation'), dataset_name], check=True)

        original_dataset_path = os.path.join(data_path_prefix, 'SlimPajama-627B')
        original_chunk_path = os.path.join(original_dataset_path, 'train', chunk_name)  # e.g., chunk1, diff chunk parallelized by diff machines

        file_list = os.listdir(original_chunk_path)
        file_list.sort()

        files_per_iteration = 50
        n_iter = len(file_list) // files_per_iteration
        file_splits = np.array_split(file_list, n_iter)

        for part_id, cur_iter_files in enumerate(file_splits):

            shutil.rmtree(os.path.join(dataset_name, 'train'), ignore_errors=True)
            os.makedirs(os.path.join(dataset_name, 'train'), exist_ok=True)
            for file in cur_iter_files:
                s_file_path = os.path.join(original_dataset_path, 'train', chunk_name, file)
                t_file_path = os.path.join(dataset_name, 'train', file)
                subprocess.run(['ln', s_file_path, t_file_path], check=True)

            os.makedirs(os.path.join(data_path_prefix, 'SlimPajama-627B-llama3-tokenized'), exist_ok=True)
            cache_dir = Path(os.path.join(data_path_prefix, f'SlimPajama-627B-llama3-tokenized/{chunk_name}_part_{part_id}'))  # path to save tokenized dataset
            num_workers = num_cpu_cores() // 2
            chunk_size = 16
            datamodule = LMDataModule(dataset_name, tokenizer_name='meta-llama/Meta-Llama-3.1-8B',
                                      dataset_config_name=None,
                                      max_length=2048, cache_dir=cache_dir,
                                      add_eos=True, batch_size=8,
                                      num_workers=num_workers, use_shmem=False,
                                      raw_json_path=None, pad_to_multiple_of=chunk_size)
            datamodule.prepare_data()

            shutil.rmtree(os.path.join(dataset_name, 'train'), ignore_errors=True)

    def test_dclm(self):
        # dataset_name = '/home/xiaolwang/new_home/datasets/DCLM-Baseline-100B-json'
        dataset_name = '/home/xinhaoli/datasets/DCLM-Baseline-100B-json-text'  # GCP machine
        # dataset_name = '/juice5/scr5/nlp/mttt/datasets/DCLM-Baseline-100B-json-processed/train'
        dataset_config_name = None
        # cache_dir = Path(f'/home/xiaolwang/new_home/datasets/DCLM-Baseline-100B-llama3-tokenized')  # path to save tokenized dataset
        # cache_dir = Path('/home/xinhaoli/datasets/DCLM-Baseline-100B-llama3-tokenized')
        cache_dir = Path('/home/xinhaoli/datasets/DCLM-Baseline-100B-llama3-tokenized-no-pod')
        # cache_dir = Path('/juice5/scr5/nlp/mttt/datasets/DCLM-Baseline-100B-llama3-tokenized')
        batch_size = 8
        max_length = 2048
        num_workers = num_cpu_cores() // 2
        # chunk_size = 16  # @xinhao: 10/31
        chunk_size = 0  # @xinhao 11/01
        # Dataset is too large to fit into memory, need to use disk for concatenation
        datamodule = LMDataModule(dataset_name, tokenizer_name='meta-llama/Meta-Llama-3.1-8B',
                                  dataset_config_name=dataset_config_name,
                                  max_length=max_length, cache_dir=cache_dir,
                                  add_eos=True, batch_size=batch_size,
                                  num_workers=num_workers, use_shmem=False,
                                  raw_json_path=None,
                                  pad_to_multiple_of=chunk_size)
        datamodule.prepare_data()

    def test_finewebedu(self):
        dataset_name = 'HuggingFaceFW/fineweb-edu'  # GCP machine
        dataset_config_name = None
        cache_dir = Path('/home/xinhaoli/datasets/FineWebEdu-100B-llama3-tokenized')
        batch_size = 8
        max_length = 2048
        num_workers = num_cpu_cores() // 2
        chunk_size = 16
        # Dataset is too large to fit into memory, need to use disk for concatenation
        datamodule = LMDataModule(dataset_name, tokenizer_name='meta-llama/Meta-Llama-3.1-8B',
                                  dataset_config_name=dataset_config_name,
                                  max_length=max_length, cache_dir=cache_dir,
                                  add_eos=True, batch_size=batch_size,
                                  num_workers=num_workers, use_shmem=False,
                                  raw_json_path=None,
                                  pad_to_multiple_of=chunk_size)
        datamodule.prepare_data()
