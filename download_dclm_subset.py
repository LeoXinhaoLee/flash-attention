import math
import random
import numpy as np

random.seed(42)
np.random.seed(42)

num_global_shards = 10
num_local_shards = 10
num_json = 279

num_total_file = num_global_shards * num_local_shards * num_json

file_abs_id = [i for i in range(num_total_file)]

sampled_abs_id = random.sample(file_abs_id, math.ceil(num_total_file / 40))  # 4T tokens -> 100B tokens

link_format = 'https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0/blob/main/global-shard_{:02}_of_10/local-shard_{}_of_10/shard_{:08}_processed.jsonl.zst?download=true'

link_list = []
for abs_id in sampled_abs_id:
    global_id = abs_id // (num_local_shards * num_json)
    local_id = (abs_id % (num_local_shards * num_json)) // num_json
    file_id = (abs_id % (num_local_shards * num_json)) % num_json
    link = link_format.format(global_id + 1, local_id, file_id)
    link_list.append(link)

with open("links.txt", "w") as f:
    for link in link_list:
        f.write(link + "\n")

print(len(link_list))
print(len(set(link_list)))

# import zstandard as zstd
# import json
# with open("/Users/xinhaoli/Desktop/shard_00000000_processed.jsonl.zst", "rb") as compressed_file:
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