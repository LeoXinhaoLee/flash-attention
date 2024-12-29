import os
import glob
import pdb
from pathlib import Path
import zstandard as zstd
import json
from tqdm import tqdm

# src_dir = Path("/home/xinhaoli/datasets/DCLM-Baseline-100B-json")
# tgt_dir = Path("/home/xinhaoli/datasets/DCLM-Baseline-100B-json-text")
src_dir = Path("/home/xinhaoli/datasets/DCLM-Baseline-25B-json")
tgt_dir = Path("/home/xinhaoli/datasets/DCLM-Baseline-25B-json-text")
tgt_dir.mkdir(parents=True, exist_ok=True)
json_list = list(src_dir.glob("*.jsonl.zst"))

for file in tqdm(json_list):
    with open(file, 'rb') as compressed_file:
        dctx = zstd.ZstdDecompressor()

        with dctx.stream_reader(compressed_file) as reader:
            decompressed_data = reader.read().decode('utf-8')
            lines = decompressed_data.splitlines()

            tgt_file = tgt_dir / file.with_suffix('').name
            with open(tgt_file, 'w') as out_file:
                for line in lines:
                    record = json.loads(line)
                    new_record = dict(text=record['text'])
                    out_file.write(json.dumps(new_record) + '\n')

# src_dir = Path("/home/xinhaoli/datasets/DCLM-Baseline-100B-json")
# tgt_dir = Path("/home/xinhaoli/datasets/DCLM-Baseline-100B-json-text")
# tgt_dir.mkdir(parents=True, exist_ok=True)
# json_list = list(src_dir.glob("*.jsonl.zst"))
#
# for file in tqdm(json_list):
#     with open(file, 'rb') as compressed_file:
#         dctx = zstd.ZstdDecompressor()
#
#         tgt_file = tgt_dir / (file.with_suffix('').name + '.zst')
#         with open(tgt_file, 'wb') as out_file:
#             cctx = zstd.ZstdCompressor()
#
#             with cctx.stream_writer(out_file) as compressor:
#
#                 with dctx.stream_reader(compressed_file) as reader:
#                     decompressed_data = reader.read().decode('utf-8')
#                     lines = decompressed_data.splitlines()
#
#                     for line in lines:
#                         record = json.loads(line)
#                         new_record = dict(text=record['text'])
#
#                         # Compress and write each JSON line to the .jsonl.zst file
#                         json_line = json.dumps(new_record) + '\n'
#                         compressor.write(json_line.encode('utf-8'))


# import os
# import json
# from pathlib import Path
# import zstandard as zstd
# from concurrent.futures import ProcessPoolExecutor
# from tqdm import tqdm
#
# src_dir = Path("/home/xinhaoli/datasets/DCLM-Baseline-100B-json")
# tgt_dir = Path("/home/xinhaoli/datasets/DCLM-Baseline-100B-json-text")
# json_list = list(src_dir.glob("*.jsonl.zst"))
#
# # Ensure target directory exists
# tgt_dir.mkdir(parents=True, exist_ok=True)
#
# # Function to decompress and write each file
# def process_file(file):
#     with open(file, 'rb') as compressed_file:
#         dctx = zstd.ZstdDecompressor()
#         with dctx.stream_reader(compressed_file) as reader:
#             buffer = ""
#             out_file_path = tgt_dir / file.with_suffix('').name  # Remove .zst suffix
#             with open(out_file_path, 'w') as out_file:
#                 while True:
#                     chunk = reader.read(65536).decode('utf-8')  # Read in chunks of 64KB
#                     if not chunk:
#                         break
#                     buffer += chunk
#                     *lines, buffer = buffer.splitlines()
#
#                     for line in lines:
#                         record = json.loads(line)
#                         new_record = {"text": record.get("text")}
#                         json.dump(new_record, out_file)
#                         out_file.write('\n')
#
#                 # Process any remaining data in buffer as the last line
#                 if buffer:
#                     record = json.loads(buffer)
#                     new_record = {"text": record.get("text")}
#                     json.dump(new_record, out_file)
#                     out_file.write('\n')
#
# num_cpus = os.cpu_count()
# with ProcessPoolExecutor(max_workers=num_cpus) as executor:
#     list(tqdm(executor.map(process_file, json_list), total=len(json_list)))
