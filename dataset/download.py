from transformers import GPT2Tokenizer
from datasets import load_dataset

dataset = load_dataset("cerebras/SlimPajama-627B", cache_dir="/home/s2497456/mnt/workdir/RetrofittingLLM/dataset/SlimPajama-627B")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

print(dataset["train"]["etag"])
print(dataset["train"]["url"])
tokenized_dataset = dataset.map(tokenize_function, batched=True)
print(tokenized_dataset)
