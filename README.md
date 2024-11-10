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
3. Then try Gemma2-2b
   ![image](https://github.com/user-attachments/assets/0aabeb76-e490-4b89-b4f5-9cd97633e055)

   The method to concate the memory, past_kv, and current_kv:
   ![image](https://github.com/user-attachments/assets/c050e85f-dc8b-4a3a-8253-107d44d923cb)



## Evaluation result
<img src="https://github.com/user-attachments/assets/1e91a2a5-f6fd-4d3c-90c9-d4d3c729d9fc" alt="image-20240902232209785" width="600" />
<br />
<img src="https://github.com/user-attachments/assets/1b24cb06-ad74-4832-a1e3-87629438a885" alt="image-2024090287629438a885" width="400" />


