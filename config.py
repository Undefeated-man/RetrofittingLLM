# pretrained model config
# batch_size = 2
# eval_batch_size = 4
# dataset = "cerebras/SlimPajama-627B"
# eval_steps = 50
# learning_rate = 5e-6
# grad_clip = 1.0
# mix_precision = True
# grad_accumulation_steps = 10
# preload = False
# weight_decay = 1e-4
# load_best_model_at_end = True
# metric_for_best_model = "eval_perplexity"
# greater_is_better = False
# total_steps = 7000
# warmup_steps = total_steps//10 if total_steps < 2000 else 200
# max_input_length = 1024
# eval_sample = 1000
# save_steps = 50
# model_name = "feedback-gemma"
# checkpoint_dir = None
# eval_dataset = None
# # checkpoint_dir = "/workspace/RetrofittingLLM/results/checkpoint-5000/"
# # eval_dataset = "CogComp/trec" # "cais/mmlu"
# tuning_mode = False

# finetuning config
tuning_mode = True
tuning_set = "stanfordnlp/coqa"  # "EleutherAI/lambada_openai"  # "cais/mmlu"
eval_dataset = "stanfordnlp/coqa"  # "EleutherAI/lambada_openai" # "cais/mmlu"
lora_r = 12
lora_alpha = 16
batch_size = 2
eval_batch_size = 16
epochs = 10
eval_steps = 50
learning_rate = 2e-5
grad_clip = 1.0
mix_precision = True
grad_accumulation_steps = 10
preload = False
weight_decay = 1e-4
load_best_model_at_end = True
metric_for_best_model = "eval_accuracy"
greater_is_better = True
total_steps = 3000
warmup_steps = total_steps//10 if total_steps < 2000 else 200
max_input_length = 512
eval_sample = 1000
model_name = "feedback-gemma"
# checkpoint_dir = None
checkpoint_dir = "/workspace/RetrofittingLLM/results/checkpoint-1550/"
save_steps = 100

# Basic config
config_dict = {
    "gpt2": "./config/gpt2.json",
    "feedback-gpt2": "./config/feedback_gpt2.json",
    "tinyllama": "./config/tinyllama.json",
    "feedback-tinyllama": "./config/tinyllama.json",
    "gemma": "./config/gemma2.json",
    "gemma2": "./config/gemma2.json",
    "feedback-gemma": "./config/gemma2.json",
}

device = "gpu"
precision = "bf16"
log_dir = "logs"
evalute_after_train = False
config_pth = config_dict[model_name] # "./config/gpt2.json" # "./config/feedback_gpt2.json" # "./config/tinyllama.json"  # None