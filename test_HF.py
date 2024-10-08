from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

text = "Hello World!"
token_ids = tokenizer(text)
print(token_ids)
