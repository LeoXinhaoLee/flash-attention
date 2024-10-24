import h5py
import numpy as np
from tqdm import tqdm


def convert_h5_to_npy(h5_file_path, npy_file_path, dataset_name, chunk_size=1000):

    with h5py.File(h5_file_path, 'r') as h5_file:
        dataset = h5_file[dataset_name]
        total_size = dataset.shape[0]  # Total number of rows
        data_shape = dataset.shape[1:]  # Shape of the data excluding the first dimension (rows)

        # Create a memory-mapped NumPy file with the correct shape
        mmap_array = np.lib.format.open_memmap(
            npy_file_path, mode='w+', dtype=dataset.dtype, shape=(total_size,) + data_shape
        )

        # Iterate over the dataset in chunks
        for i in tqdm(range(0, total_size, chunk_size)):
            end = min(i + chunk_size, total_size)
            mmap_array[i:end] = dataset[i:end]  # Load the chunk directly into the memory-mapped file

        # Ensure the changes are written to disk
        del mmap_array

    print(f"Data successfully converted to {npy_file_path}.")


# Example usage
h5_file_path = '/juice5/scr5/nlp/mttt/datasets/SlimPajama-627B-llama3-tokenized-100B-below-100K/tokenizer_name-meta-llama/Meta-Llama-3.1-8B-val_ratio-0.0005-val_split_seed-2357-add_eos-True-detokenize-False/train.h5'  # Replace with your HDF5 file path
npy_file_path = '/juice5/scr5/nlp/mttt/datasets/SlimPajama-627B-llama3-tokenized-100B-below-100K-npy/tokenizer_name-meta-llama/Meta-Llama-3.1-8B-val_ratio-0.0005-val_split_seed-2357-add_eos-True-detokenize-False/train.npy'  # Replace with the desired output file path

# h5_file_path = '/juice5/scr5/nlp/mttt/datasets/SlimPajama-627B-llama3-tokenized-100B-below-100K/tokenizer_name-meta-llama/Meta-Llama-3.1-8B-val_ratio-0.0005-val_split_seed-2357-add_eos-True-detokenize-False/train_subset.h5'  # Replace with your HDF5 file path
# npy_file_path = '/juice5/scr5/nlp/mttt/datasets/SlimPajama-627B-llama3-tokenized-100B-below-100K/tokenizer_name-meta-llama/Meta-Llama-3.1-8B-val_ratio-0.0005-val_split_seed-2357-add_eos-True-detokenize-False/train_converted.npy'  # Replace with the desired output file path

dataset_name = 'train_dataset'  # Replace with the dataset name inside the HDF5 file

convert_h5_to_npy(h5_file_path, npy_file_path, dataset_name, chunk_size=1000000)  # 4 bytes * 1e6: 4M
