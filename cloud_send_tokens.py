import pdb
from pathlib import Path
from google.cloud import storage

# Initialize Google Cloud Storage client
client = storage.Client()

# Define the destination bucket
bucket_name = 'xinhao-dev'
bucket = client.bucket(bucket_name)

# List all files in the bucket
# blobs = bucket.list_blobs()
# for blob in blobs:
#     print(blob.name)

# Define the directory path
directory = Path("/juice5/scr5/nlp/mttt/datasets/SlimPajama-627B-llama3-tokenized")

keywd = "10K_100K"
npy_files = directory.rglob(f"*{keywd}*.npy")
for file_path in npy_files:
    pdb.set_trace()
    blob = bucket.blob(f'SlimPajama-627B-llama3-tokenized/{keywd}/{file_path.name}')  # Destination path in the bucket
    blob.upload_from_filename(str(file_path))
    print(f'Uploaded {file_path} to gs://{bucket_name}/SlimPajama-627B-llama3-tokenized/{keywd}/{file_path.name}')
