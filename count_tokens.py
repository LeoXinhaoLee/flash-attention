import pdb
import os
import os.path as osp
from pathlib import Path
import glob
import numpy as np

directory = Path("/juice5/scr5/nlp/mttt/datasets/SlimPajama-627B-llama3-tokenized")

npy_files = directory.rglob("*below_10K*.npy")
# npy_files = directory.rglob("*10K_100K*.npy")
# npy_files = directory.rglob("*100K_200K*.npy")
# npy_files = directory.rglob("*200K_500K*.npy")
# npy_files = directory.rglob("*500K_1M*.npy")
# npy_files = directory.rglob("*above_1M*.npy")

token_count = 0

for npy_file in npy_files:
    # print(npy_file)
    data = np.load(npy_file, mmap_mode='r')
    token_count += len(data)

print(f'Total token count: {token_count}')
