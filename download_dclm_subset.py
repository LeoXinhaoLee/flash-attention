import math
import random
import numpy as np

# random.seed(42)
# np.random.seed(42)

random.seed(0)
np.random.seed(0)

num_global_shards = 10
num_local_shards = 10
num_json = 279

num_total_file = num_global_shards * num_local_shards * num_json

file_abs_id = [i for i in range(num_total_file)]

# sampled_abs_id = random.sample(file_abs_id, math.ceil(num_total_file / 40))  # 4T tokens -> 100B tokens
sampled_abs_id = random.sample(file_abs_id, math.ceil(num_total_file / 160))  # 4T tokens -> 25B tokens

link_format = 'https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0/resolve/main/global-shard_{:02}_of_10/local-shard_{}_of_10/shard_{:08}_processed.jsonl.zst?download=true'

f = open('links.txt')
existing_links = f.readlines()
f.close()

link_list = []
for abs_id in sampled_abs_id:
    global_id = abs_id // (num_local_shards * num_json)
    local_id = (abs_id % (num_local_shards * num_json)) // num_json
    file_id = (abs_id % (num_local_shards * num_json)) % num_json
    link = link_format.format(global_id + 1, local_id, file_id)
    if link not in existing_links:
        link_list.append(link)

with open("links_2.txt", "w") as f:
    for link in link_list:
        f.write(link + "\n")

print(len(link_list))
print(len(set(link_list)))

# import zstandard as zstd
# import json
# with open("/juice5/scr5/nlp/mttt/datasets/DCLM-Baseline-100B-json/train/global-shard_03_of_10_local-shard_3_of_10_shard_00000098_processed.jsonl.zst",
#           "rb") as compressed_file:
#     dctx = zstd.ZstdDecompressor()
#     with dctx.stream_reader(compressed_file) as reader:
#         # Read entire decompressed content at once
#         decompressed_data = reader.read().decode('utf-8')
#
#         # Split the content into lines
#         lines = decompressed_data.splitlines()
#
#         # Parse each line as JSON
#         for line in lines:
#             line_content = json.loads(line)
#             print(line_content)

# import zstandard as zstd
# import json
#
# with open('/juice5/scr5/nlp/mttt/datasets/DCLM-Baseline-100B-json/train/global-shard_01_of_10_local-shard_2_of_10_shard_00000261_processed.jsonl.zst', 'rb') as compressed_file:
#     dctx = zstd.ZstdDecompressor()
#     with dctx.stream_reader(compressed_file) as reader:
#         decompressed_data = reader.read().decode('utf-8')
#         lines = decompressed_data.splitlines()
#         with open('/juice5/scr5/nlp/mttt/datasets/DCLM-Baseline-100B-json-processed/train/shard_00000058_processed.jsonl', 'w') as out_file:
#             for line in lines:
#                 record = json.loads(line)
#                 record.pop('metadata')
#                 # if "metadata" in record:
#                 #     record["metadata"].setdefault("WARC-Truncated", None)
#
#                 out_file.write(json.dumps(record) + '\n')
