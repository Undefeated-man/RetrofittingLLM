# 用MMLU数据集测试RetrofittingLLM模型的性能
from datasets import load_dataset
from transformerlib import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from utils import *

import config as args


tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = get_model(args)
dataset = load_dataset(args.eval_dataset, streaming=True)


inputs = tokenizer("Tom is", return_tensors='pt')
generated = model.generate(**inputs, max_new_tokens=100)[0]
print(tokenizer.decode(generated)) 