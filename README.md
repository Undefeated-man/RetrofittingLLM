# Retrofitting LLMs into feedback transformers

------

## To run 

## Current Issues Tracking
1. (Solved) I used the "\n" to split each sample, and set the batch size as 8 (max_len=512), but still got CUDA_OUT_OF_MEMORY. $\Rightarrow$ Trying to reduce the batch size, use DP training, and utilize tricks like gradient accumulation and mix precision training.
   
2. (Solved) Packages version conflict (transformers==4.42.4).
------

## Datasets and Tasks

### Pre-training:

[SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B?row=1) - Training the Model on this dataset (50-200)



### Evaluation (downstream tasks):

[MMLU](https://huggingface.co/datasets/cais/mmlu) - a benchmark to evaluate the model
[CoQA](https://huggingface.co/datasets/stanfordnlp/coqa) - a question answering task benchmark to evaluate the model
[LAMBADA](https://huggingface.co/datasets/cimec/lambada) - to test the long-range dependencies



## Models

1. Start from GPT2
2. Then try TinyLlama v1.1 (1.1B)
3. Then try gemma2-2b

