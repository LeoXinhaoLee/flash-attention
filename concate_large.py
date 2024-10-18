import pdb
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os


def concatenate_npy_files(file_list, output_file):
    # Get the shape and dtype of the first file
    sample_file = np.load(file_list[0], mmap_mode='r')
    dtype = sample_file.dtype
    shape = sample_file.shape

    # Calculate total number of rows for the concatenated file
    total_rows = sum(np.load(file, mmap_mode='r').shape[0] for file in file_list)

    # Preallocate the output file
    concatenated_array = np.memmap(output_file, dtype=dtype, mode='w+', shape=(total_rows,))

    # Start concatenation process
    current_index = 0
    for file in tqdm(file_list):
        data = np.load(file, mmap_mode='r')
        rows = data.shape[0]

        # Copy data from current file to the appropriate slice in the output file
        concatenated_array[current_index:current_index + rows] = data
        current_index += rows

    # Flush the changes and close the memmap file
    print('Flushing...')
    concatenated_array.flush()

directory = Path("/juice5/scr5/nlp/mttt/datasets/SlimPajama-627B-llama3-tokenized-100B-above-100K")
file_list = list(directory.rglob("*.npy"))
file_list = [file for file in file_list if "500K_1M" in file.name or "above_1M" in file.name]
output_file = directory / "train.npy"

concatenate_npy_files(file_list, output_file)
