# Retrofitting LLMs into feedback transformers

------

## Current Issues Tracking
1. (Solved) I used the "\n" to split each sample, and set the batch size as 8 (max_len=512), but still got CUDA_OUT_OF_MEMORY. $\Rightarrow$ Trying to reduce the batch size, use DP training, and utilize tricks like gradient accumulation and mix precision training.
   
2. Packages version conflict (transformers==4.35.0).
------

## Datasets and Tasks

### Pre-training:

[SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B?row=1) - Training the Model on this dataset (50-200)



### Evaluation (downstream tasks):

[MMLU](https://github.com/hendrycks/test) - a benchmark to evaluate the model

1. [TREC](https://huggingface.co/datasets/trec) - Test the memory and retrieval capacity.
2. [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) - Test how performance improves when  the memory improves.
3. [GSM8k](https://huggingface.co/datasets/gsm8k) - Evaluate the model's multihops reasoning capacity.
4. [MultiWoz](https://github.com/budzianowski/multiwoz) - Access its multihops reasoning capacity on multiturns conversation (long context, without positional bias)
5. [Scroll benchmark](https://www.scrolls-benchmark.com/getting-started) - Evaluate the overall performance on QA, Summarisation, NL Inference tasks over multiple domains with long context.





## Models

1. Start from GPT2
2. Then try gemma 2b
3. Then try llama2 7b (not sure if the GPU is enough)

## Results can be viewed here (To be continued...)
Online result table: https://1drv.ms/x/c/402ef993809fda1e/EYvDxtt2SAZHjNa5OaGoZ-0BeAkV_OqgAsqu-e27EJIIjQ?e=ZwHhs3
