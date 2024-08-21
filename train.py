from transformerlib import TrainerCallback, AutoTokenizer, set_seed, AdamW, DataCollatorWithPadding, DataCollatorForLanguageModeling, Trainer, TrainingArguments, TrainerState
from torch.utils.data import DataLoader
from datasets.iterable_dataset import IterableDataset
from datasets import load_dataset, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.nn import DataParallel
from torch.cuda import device_count
from collections import OrderedDict
from functools import wraps
# from accelerate import Accelerator
from numpy import mean, exp
from torch import cuda, nn
from utils import *

import numpy as np
import config as args
import datetime
import tqdm
import torch
import os
import time
import gc


def check_amp_available():
    """
        Check if Automatic Mixed Precision is available
    """
    if not torch.cuda.is_available():
        return False
    try:
        torch.cuda.amp.autocast(enabled=True)
        return True
    except:
        return False

def load_weights(model, pth, model_name):
    """
        Load the trained model weights.
    """
    state_dict = torch.load(pth)[model_name]["model_state_dict"]
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # 去除`module.`前缀
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def perplexity(logits, labels):
    """
        Calculate the perplexity
        
        Args:
            loss: 
    """
    return torch.exp(-torch.log(torch.gather(logits, 1, labels.unsqueeze(1))).mean())


def train(model, dataloader, optimizer, test_dataloader=None, num_epochs=3, accumulation_steps=1, scaler=None, load_weights_pth=None):
    """
        Train the model
        
        Args:
            model: nn.Module, the model to train
            dataloader: DataLoader, the dataloader for the train dataset
            test_dataloader: DataLoader, the dataloader for the test dataset
            optimizer: torch.optim, the optimizer for the model
            num_epochs: int, the number of epochs
    """
    
    # load the weights
    if load_weights_pth:
        load_weights(model, load_weights_pth)
    
    if device_count() > 1:
        model = DataParallel(model).cuda()
    elif device_count() == 1:
        model = model.cuda()
    model.train()
    best_perplexity = float('inf')
    stop_cnt = 0
    amp_available = check_amp_available()
    for epoch in tqdm.trange(num_epochs):
        if stop_cnt > 3:  # early stop
            print("Early stop! Now the epoch is: ", epoch)
            break
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            if test_dataloader and (batch_idx + 1) % 100 == 0:
                tmp_perx = evaluate(model, test_dataloader)
                if tmp_perx < best_perplexity:
                    best_perplexity = tmp_perx
                    stop_cnt = 0
                    torch.save(
                        {
                            "gpt2":{
                                'epoch': epoch,
                                'model_state_dict': model.state_dict()
                            }
                        }, f"output/{model.__class__.__name__}.params"
                    )
                else:
                    stop_cnt += 1
            inputs = {key: value.to(model.device) for key, value in batch.items()}
            if scaler and amp_available:
                with autocast():
                    outputs = model(**inputs)
                    loss = outputs.loss / accumulation_steps
                    scaler.scale(loss).backward()
            else:
                outputs = model(**inputs)
                loss = outputs.loss / accumulation_steps
                loss.backward()
                
            if (batch_idx+ 1) % accumulation_steps == 0:
                if scaler and amp_available:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")
            
            
def evaluate(model, dataloader, load_weights_pth=None):
    """
        Evaluate the model
        
        Args:
            model: nn.Module, the model to evaluate
            dataloader: DataLoader, the dataloader for the dataset
            
        Returns:
            float, the perplexity
    """
    # load the weights
    if load_weights_pth:
        load_weights(model, load_weights_pth)
    
    if device_count() > 1:
        model = DataParallel(model).cuda()
    elif device_count() == 1:
        model = model.cuda()
    model.eval()
    amp_available = check_amp_available()
    perx_ls = []
    with tqdm.tqdm(total=len(dataloader)) as pbar:
        for idx, batch in enumerate(dataloader):
            inputs = {key: value.to(model.device) for key, value in batch.items()}
            if amp_available:
                with torch.no_grad():
                    with autocast():
                        out = model(**inputs)
            else:
                with torch.no_grad():
                    out = model(**inputs)
            # outputs_list += out.logits
            # labels_list += inputs['labels']
            perx_ls.append(out.loss.item())
            pbar.update(1)
            if idx > 100:
                pbar.update(len(dataloader) - 100)
                break
    perx = exp(mean(perx_ls))
    print(f"\nThe Perplexity: {perx}\n")
    return perx
    

def slicing_with_window(dicts, window_size, step):
    res = []
    for i in range(0, dicts["input_ids"].shape[1] - window_size + 1, step):
        slice_dict = {}
        for key, value in dicts.items():
            slice_dict[key] = value[:, i:i+window_size]
        res.append(slice_dict)
    return res


class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # 自定义日志记录逻辑
        now = datetime.datetime.now()
        formatted_date = now.strftime("%Y-%m-%d")
        try:
            logs["train_perplexity"] = exp(logs["loss"])
        except Exception as e:
            print("Error: %s"%e)

        try:
            logs["eval_perplexity"] = exp(logs["eval_loss"])
        except Exception as e:
            print("Error: %s"%e)
        print(logs)
        with open(f"./logs/log_{formatted_date}_{model_name}", "a") as f:
            f.write(str(logs) + "\n")
        


if args.tuning_mode:
    if args.eval_dataset == "stanfordnlp/coqa":
        class CustomTrainer(Trainer):
            def save_model(self, output_dir=None, _internal_call=False):
                if output_dir is None:
                    output_dir = self.args.output_dir

                # self.model.lm_head.weight = torch.nn.Parameter(self.model.transformer.wte.weight.clone())
                self.model.save_pretrained(output_dir, safe_serialization=False)
                torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
                self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
                # super().save_model(f"{output_dir}/{self.model.__class__.__name__}.param", _internal_call=_internal_call)
                # self.model.lm_head.weight = self.model.transformer.wte.weight
                
            def compute_loss(self, model, inputs, return_outputs=False):
                """For QA task"""
                start_positions = inputs.pop("start_positions")
                end_positions = inputs.pop("end_positions")
                
                # 获取模型输出
                outputs = model(**inputs)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                
                # 计算损失
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2
                return (loss, outputs) if return_outputs else loss
            
    elif args.eval_dataset == "EleutherAI/lambada_openai":
        class CustomTrainer(Trainer):
            def save_model(self, output_dir=None, _internal_call=False):
                if output_dir is None:
                    output_dir = self.args.output_dir

                # self.model.lm_head.weight = torch.nn.Parameter(self.model.transformer.wte.weight.clone())
                self.model.save_pretrained(output_dir, safe_serialization=False)
                torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
                self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
                # super().save_model(f"{output_dir}/{self.model.__class__.__name__}.param", _internal_call=_internal_call)
                # self.model.lm_head.weight = self.model.transformer.wte.weight

            def compute_loss(self, model, inputs, return_outputs=False):
                # 获取输入
                labels = inputs.pop("labels")
                # 前向传播
                outputs = model(**inputs)
                logits = outputs.logits
    
                # 获取最后一个token的预测结果
                # predicted_token_id = torch.argmax(logits[:, -1, :], dim=-1)
                # 计算损失
                # print(predicted_token_id)
                loss = nn.CrossEntropyLoss()(logits[torch.arange(logits.size(0)), labels[:, 1].view(-1), :], labels[:, 0].view(-1))
                return (loss, outputs) if return_outputs else loss
    else:
        class CustomTrainer(Trainer):
            def save_model(self, output_dir=None, _internal_call=False):
                if output_dir is None:
                    output_dir = self.args.output_dir

                # self.model.lm_head.weight = torch.nn.Parameter(self.model.transformer.wte.weight.clone())
                torch.save(model, f"{model.__class__.__name__}.model")
                self.model.save_pretrained(output_dir, safe_serialization=False)
                torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
                self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
else:
    if "feedback" in args.model_name:
        class CustomTrainer(Trainer):
            def save_model(self, output_dir=None, _internal_call=False):
                if output_dir is None:
                    output_dir = self.args.output_dir

                # self.model.lm_head.weight = torch.nn.Parameter(self.model.transformer.wte.weight.clone())
                torch.save(model, f"{model.__class__.__name__}.model")
                self.model.save_pretrained(output_dir, safe_serialization=False)
                torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
                self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
                # super().save_model(f"{output_dir}/{self.model.__class__.__name__}.param", _internal_call=_internal_call)
                # self.model.lm_head.weight = self.model.transformer.wte.weight
            
            # def compute_loss(self, model, inputs, return_outputs=False):
            #     # 获取模型输入并传递past_key_values
            #     memory = None
            #     past_key_values = None
            #     loss = None
            #     slice_len = args.max_input_length//10
                
            #     for input in slicing_with_window(inputs, slice_len, slice_len//2):
            #         # print(input)
            #         if memory is not None:
            #             input['memory'] = memory
            #             input['past_key_values'] = past_key_values
            #         # 模型前向传播
            #         # print(f"loss before: {loss}")
            #         outputs = model(**input)
            #         # 更新past_key_values
            #         memory = outputs.memory
            #         past_key_values = outputs.past_key_values
            #         # compute loss
            #         if loss is None:
            #             loss = outputs.loss
            #         else:
            #             loss += outputs.loss
            #         # print(f"loss after: {loss}")
            #     return (loss, outputs) if return_outputs else loss
    else:
        class CustomTrainer(Trainer):
            def save_model(self, output_dir=None, _internal_call=False):
                if output_dir is None:
                    output_dir = self.args.output_dir

                # self.model.lm_head.weight = torch.nn.Parameter(self.model.transformer.wte.weight.clone())
                torch.save(model, f"{model.__class__.__name__}.model")
                self.model.save_pretrained(output_dir, safe_serialization=False)
                torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
                self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
                # super().save_model(f"{output_dir}/{self.model.__class__.__name__}.param", _internal_call=_internal_call)
                # self.model.lm_head.weight = self.model.transformer.wte.weight
    
    # def training_step(self, model, inputs):
    #     model.train()
    #     inputs = self._prepare_inputs(inputs)
    #     start_positions = inputs.pop("start_positions")
    #     end_positions = inputs.pop("end_positions")
    #     outputs = model(**inputs)
    #     start_logits = outputs.start_logits
    #     end_logits = outputs.end_logits

    #     # Compute the loss using start and end positions
    #     loss_fct = torch.nn.CrossEntropyLoss()
    #     start_loss = loss_fct(start_logits, start_positions)
    #     end_loss = loss_fct(end_logits, end_positions)
    #     loss = (start_loss + end_loss) / 2

    #     loss.backward()
    #     return loss.detach()


def pad_sequences_with_lengths(sequences, max_len, padding_value=-100):
    # 保存每个序列的有效长度
    lengths = torch.tensor(len(sequences))
    sequences = torch.tensor(sequences)
    # 初始化填充后的张量
    padded_sequences = torch.full((max_len,), padding_value, dtype=lengths.dtype)
    # 进行填充
    padded_sequences[:lengths] = sequences
    return padded_sequences, lengths


def process_with_retry(func, retries=3, delay=60, **kwargs):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)  # 等待一段时间再重试
                else:
                    raise e  # 超过最大重试次数，抛出异常
    return wrapper


@process_with_retry
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=args.max_input_length)


def sliding_window(text, max_length=1024, overlap=50):
    tokens = tokenizer.encode(text['text'])
    stride = max_length - overlap
    token_blocks = [tokens[i:i+max_length] for i in range(0, len(tokens), stride) if i+max_length <= len(tokens)]
    return token_blocks


@process_with_retry
def prepare_mmlu(samples, batch_size):
    res = []
    ans = []
    choices = ["A", "B", "C", "D"]
    batch_index = 0
    for question, c, answer in zip(samples["question"], samples["choices"], samples["answer"]):
        choice = "; ".join(f"{i}. {j}" for i, j in zip(choices, c))
        sample = f"<question>{question}</question><choice>{choice}</choice><answer>"
        res.append(sample)
        ans.append(choices[answer])
    
    while batch_index < len(res):
        yield tokenizer(res[batch_index:batch_index+batch_size], return_tensors='pt', truncation=True, \
            padding='max_length', max_length=args.max_input_length), ans[batch_index:batch_index+batch_size]
        batch_index += batch_size


@process_with_retry
def preprocess_cl_function(examples):
    inputs = []
    labels = []
    
    for text in examples['text']:
        # 分割文本，将最后一个词作为标签
        split_text = text.split()
        if len(split_text) < 2:
            continue
        
        input_text = " ".join(split_text[:-1])
        label_text = split_text[-1]
        
        # 对输入和标签分别进行分词
        input_ids = tokenizer(input_text, truncation=True, max_length=args.max_input_length)["input_ids"]
        input_ids, length = pad_sequences_with_lengths(input_ids, args.max_input_length, padding_value=tokenizer.pad_token_id)
        label_id = tokenizer(label_text, add_special_tokens=False)["input_ids"][0] # use the only first token
        label_id = torch.tensor([label_id, length])
        
        inputs.append(input_ids)
        labels.append(label_id)
    
    # 返回一个字典，包含输入和标签
    return {"input_ids": inputs, "labels": labels}


@process_with_retry
def instruct(samples):
    """
        Instruct the model to generate the text
        
        Args:
            samples: list, the samples
        
        Returns:
            list, the generated text
    """
    res = []
    choices = ["A", "B", "C", "D"]
    for question, c, answer in zip(samples["question"], samples["choices"], samples["answer"]):
        choice = "; ".join(f"{i}. {j}" for i, j in zip(choices, c))
        sample = f"<question>{question}</question><choice>{choice}</choice><answer>{choices[answer]}</answer>"
        res.append(sample)
        # res.append(tokenizer(sample, return_tensors='pt'))
        # res = {key: value.to(device) for key, value in inputs.items()}
    return tokenizer(res, return_tensors='pt', truncation=True, \
        padding='max_length', max_length=args.max_input_length)


def get_subset(iterable_dataset, num_samples):
    subset_data = []
    for i, sample in enumerate(iterable_dataset):
        if i >= num_samples:
            break
        subset_data.append(sample["text"])
    return subset_data


def clean(text):
    text = text.replace("</answer>", "")
    text = text.replace("</", "")
    return text.strip()


@process_with_retry
def prepare_qa_dataset(examples):
    contexts = examples['story']
    questions_list = examples['questions']
    answers_list = examples['answers']

    input_ids = []
    attention_masks = []
    start_positions = []
    end_positions = []

    with tqdm.tqdm(total=len(contexts)) as pbar:
        for context, questions, answers in zip(contexts, questions_list, answers_list):
            pbar.update(1)
            for question, start, end in zip(questions, answers['answer_start'], answers['answer_end']):
                # Tokenize the pair of context and question
                inputs = tokenizer(question, context, truncation=True, padding="max_length", \
                    max_length=args.max_input_length - 50, return_offsets_mapping=True)
                
                input_ids.append(inputs['input_ids'])
                attention_masks.append(inputs['attention_mask'])

                # Determine the start and end positions
                offset_mapping = inputs['offset_mapping']
                start_position = None
                end_position = None
                
                for idx, (start_offset, end_offset) in enumerate(offset_mapping):
                    if start_offset <= start < end_offset:
                        start_position = idx
                    if start_offset < end <= end_offset:
                        end_position = idx
                        break

                if start_position is not None and end_position is not None:
                    start_positions.append(start_position)
                    end_positions.append(end_position)
                else:
                    start_positions.append(0)
                    end_positions.append(0)
    
    output = {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'start_positions': start_positions,
        'end_positions': end_positions
    }
    
    return output


def compute_acc(pred):
    labels = pred.label_ids
    t = pred.predictions
    if type(t) == tuple:
        t = pred.predictions[0]
    t = t[range(t.shape[0]), labels[:, 1], :]
    preds = t.argmax(-1)
    # print(t.shape)
    # if len(t.shape) == 3:
    #     preds = t[:, -1, :].argmax(-1)
    # else:
    #     preds = pred.predictions[:, -1, :].argmax(-1)
    # # preds = t[:, -1, :].argmax(-1)
    
    # 去除填充部分的影响
    # if len(t.shape) == 3:
    #     mask = labels != -100
    #     labels = labels[mask]
    #     preds = preds[mask]
    
    # 计算准确率
    # print(labels, preds)
    accuracy = (preds == labels[:, 0]).astype(np.float32).mean().item()
    return {"accuracy": accuracy}


def compute_qa_metrics(p):
    start_preds = np.argmax(p.predictions[0], axis=1)
    end_preds = np.argmax(p.predictions[1], axis=1)
    start_labels = p.label_ids[0]
    end_labels = p.label_ids[1]

    start_accuracy = np.mean(start_preds == start_labels)
    end_accuracy = np.mean(end_preds == end_labels)

    return {
        "start_accuracy": start_accuracy,
        "end_accuracy": end_accuracy,
        "accuracy": (start_accuracy + end_accuracy) / 2,
    }

class SkipIterableDataset(IterableDataset):
    def __init__(self, dataset, skip_n=None):
        self.dataset = dataset
        self.__iter__(skip_n=skip_n)

    def __iter__(self, skip_n=None):
        iterator = iter(self.dataset)
        if not skip_n is None:
            for _ in range(skip_n):
                next(iterator, None)  # 跳过前 n 个样本
        return iterator

class FixDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False, mlm_probability=0.15, return_tensors="pt"):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
        self.return_tensors = return_tensors
        
    def __call__(self, examples):
        batch = super().__call__(examples)
        batch['input_ids'][:, -1:] = tokenizer.pad_token_id
        batch['labels'] = torch.cat((batch['labels'][:, 1:], batch['labels'][:, :1]), dim=-1)
        batch['labels'][:, -1:] = tokenizer.pad_token_id
        return batch


# def prepare_qa_dataset(examples):
#     context = examples['story']
#     questions_list = examples['questions']
#     answers_list = examples['answers']

#     input_ids = []
#     attention_masks = []
#     start_positions = []
#     end_positions = []

#     for question, start, end in zip(questions_list, answers_list['answer_start'], answers_list['answer_end']):
#         # Tokenize the pair of context and question
#         inputs = tokenizer(question, context, truncation=True, padding="max_length", \
#             max_length=args.max_input_length - 50, return_offsets_mapping=True)
        
#         input_ids.append(inputs['input_ids'])
#         attention_masks.append(inputs['attention_mask'])

#         # Determine the start and end positions
#         offset_mapping = inputs['offset_mapping']
#         start_position = None
#         end_position = None
        
#         for idx, (start_offset, end_offset) in enumerate(offset_mapping):
#             if start_offset <= start < end_offset:
#                 start_position = idx
#             if start_offset < end <= end_offset:
#                 end_position = idx
#                 break

#         if start_position is not None and end_position is not None:
#             start_positions.append(start_position)
#             end_positions.append(end_position)
#         else:
#             start_positions.append(0)
#             end_positions.append(0)
    
#     output = {
#         'input_ids': input_ids,
#         'attention_mask': attention_masks,
#         'start_positions': start_positions,
#         'end_positions': end_positions
#     }
#     print(len(input_ids), len(attention_masks), len(start_positions), len(end_positions))
#     return output


if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name) #, cache_dir=cache_dir)
    tokenizer, model = get_model(args)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_name = args.model_name
    
    try:
        if args.lora_r:
            from peft import LoRAConfig, get_peft_model
            
            lora_config = LoRAConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.1,
                target_modules=['attn', 'mlp']  # 根据你的模型结构选择适当的模块
            )
            model = get_peft_model(model, lora_config)
    except:
        pass
    
    set_seed(37)

    #############################
    if args.tuning_mode:
        print(f"\nTuning on: {args.tuning_set}")
        if args.eval_dataset == "stanfordnlp/coqa":
            dataset = load_dataset(args.tuning_set)
            # split the dataset into train and validation
            # train_idx = int(len(dataset) * 0.9)
            train_dataset = dataset["train"]  # [:train_idx]  auxiliary_train
            valid_dataset = dataset["validation"][:args.eval_sample]  # [train_idx:]
            print(f"\n\nGetting the first {args.eval_sample} samples for evaluation.\n\n")
            # train_dataset = Dataset.from_dict(train_dataset)
            # valid_dataset = Dataset.from_dict(valid_dataset)
            
            try:
                tokenized_train_dataset = Dataset.from_dict(prepare_qa_dataset(train_dataset))
                tokenized_valid_dataset = Dataset.from_dict(prepare_qa_dataset(valid_dataset))
            except:
                tokenized_train_dataset = train_dataset.map(prepare_qa_dataset)
                tokenized_valid_dataset = valid_dataset.map(prepare_qa_dataset)
            print(tokenized_valid_dataset)
        elif args.eval_dataset == "EleutherAI/lambada_openai":
            dataset = load_dataset(args.tuning_set, trust_remote_code=True, split="test").shuffle(seed=37)
            # split the dataset into train and validation
            # train_idx = int(len(dataset) * 0.9)
            dataset = dataset.train_test_split(test_size=0.2, seed=37)
            train_dataset = dataset["train"]
            valid_dataset = dataset["test"][:args.eval_sample]
            valid_dataset = Dataset.from_dict(valid_dataset)
            
            tokenized_train_dataset = train_dataset.map(preprocess_cl_function, batched=True)
            tokenized_valid_dataset = valid_dataset.map(preprocess_cl_function, batched=True)
            tokenized_train_dataset = tokenized_train_dataset.map(lambda x: tokenizer.pad(x, padding="max_length"), batched=True)
            tokenized_valid_dataset = tokenized_valid_dataset.map(lambda x: tokenizer.pad(x, padding="max_length"), batched=True)
        elif args.eval_dataset == "cais/mmlu":
            dataset = load_dataset(args.tuning_set, "all")
            train_dataset = dataset["auxiliary_train"] #load_dataset(args.dataset, split="train", streaming=True)
            valid_dataset = dataset["validation"].select(range(args.eval_sample)) # load_dataset(args.dataset, split="validation", streaming=True)
            tokenized_train_dataset = train_dataset.map(instruct, batched=True)
            tokenized_valid_dataset = valid_dataset.map(instruct, batched=True)
    else:
        print(f"\nPretraining on: {args.dataset}")
        dataset = load_dataset(args.dataset, streaming=True)
        train_dataset = dataset["train"] #load_dataset(args.dataset, split="train", streaming=True)
        valid_dataset = dataset["validation"] # load_dataset(args.dataset, split="validation", streaming=True)
        valid_dataset = Dataset.from_dict({"text": get_subset(valid_dataset, args.eval_sample)})
        tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
        tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True)
    
    if args.tuning_mode:
        if args.eval_dataset == "stanfordnlp/coqa":
            data_collator = DataCollatorWithPadding(tokenizer)
        elif args.eval_dataset == "EleutherAI/lambada_openai":
            data_collator = None
        elif args.eval_dataset == "cais/mmlu":
            data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    else:
        # data_collator = FixDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
    print("Process the dataset successfully!")
    training_args = TrainingArguments(
                    output_dir='./results',          # 输出目录
                    evaluation_strategy="steps",
                    lr_scheduler_type="cosine",
                    eval_steps=args.eval_steps,
                    logging_steps=args.eval_steps,
                    logging_first_step=True,
                    overwrite_output_dir=True,
                    per_device_train_batch_size=args.batch_size,   # 每个设备的批次大小
                    per_device_eval_batch_size=args.eval_batch_size,
                    save_total_limit=10,              # 保存模型的总数限制
                    logging_dir=args.log_dir,
                    learning_rate=args.learning_rate,
                    warmup_steps=args.warmup_steps,
                    max_grad_norm=args.grad_clip,
                    weight_decay=args.weight_decay,
                    fp16=args.mix_precision,
                    max_steps=args.total_steps,
                #   lr_scheduler_kwargs = args.lr_scheduler_kwargs,
                    gradient_accumulation_steps=args.grad_accumulation_steps,
                    load_best_model_at_end=args.load_best_model_at_end,
                    metric_for_best_model=args.metric_for_best_model,  # 用于选择最佳模型的指标
                    greater_is_better=args.greater_is_better,  # 指标是否越大越好
                    save_steps = args.save_steps
                )
    
    # print(tokenized_valid_dataset)
    if args.tuning_mode:
        if args.eval_dataset == "stanfordnlp/coqa":
            trainer = CustomTrainer(
                            model=model,
                            args=training_args,
                            data_collator=data_collator,
                            train_dataset=tokenized_train_dataset,
                            eval_dataset=tokenized_valid_dataset,
                            tokenizer=tokenizer,
                            compute_metrics=compute_qa_metrics,
                            callbacks=[CustomCallback()]
                        )
        elif args.eval_dataset == "EleutherAI/lambada_openai":
            trainer = CustomTrainer(
                            model=model,
                            args=training_args,
                            data_collator=data_collator,
                            train_dataset=tokenized_train_dataset,
                            eval_dataset=tokenized_valid_dataset,
                            tokenizer=tokenizer,
                            compute_metrics=compute_acc,
                            callbacks=[CustomCallback()]
                        )
        else:
            trainer = CustomTrainer(
                            model=model,
                            args=training_args,
                            data_collator=data_collator,
                            train_dataset=tokenized_train_dataset,
                            eval_dataset=tokenized_valid_dataset,
                            tokenizer=tokenizer,
                            callbacks=[CustomCallback()]
                        )
    else:
        trainer = CustomTrainer(
                            model=model,
                            args=training_args,
                            data_collator=data_collator,
                            train_dataset=tokenized_train_dataset,
                            eval_dataset=tokenized_valid_dataset,
                            tokenizer=tokenizer,
                            callbacks=[CustomCallback()]
                        )
    
    # if args.checkpoint_dir:
    #     trainer_state = TrainerState.load_from_json(os.path.join(args.checkpoint_dir, 'trainer_state.json'))
    #     trainer.state = trainer_state
    
    # trainer.train()
    # try:
    #     trainer.train()
    #     trainer.save_model()
    # except Exception as e:
    #    print("Error: %s"%e)
    #    trainer.save_model()
    # trainer.evaluate()
    # trainer.save_model()
    ##########################################################
    
    # ----------------------- unit test ------------------------ #
    # dataset = DecodersDataset(texts, tokenizer, max_length=512, test=True)
    # train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # val_dataloader = None
    # ---------------------------------------------------------- #
    
    if args.evalute_after_train:
        # Output an example to see the result
        device = torch.device("cuda")
        if args.tuning_mode:
            from sklearn.metrics import accuracy_score
            pred = []
            gt = []
            cnt = 0
            choices = ["A", "B", "C", "D"]
            with tqdm.tqdm(total=round(len(valid_dataset)/args.batch_size/6)) as pbar:
                model.to(device)
                model.eval()
                for samples, answers in prepare_mmlu(valid_dataset, args.batch_size*6):
                    samples = {key: value.to(device) for key, value in samples.items()}
                    generated = model.generate(**samples, repetition_penalty=1.2, max_new_tokens=5)
                    for sample, answer in zip(generated, answers):
                        p = tokenizer.decode(sample.cpu().numpy(), skip_special_tokens=True).replace("\n", " ")
                        if "<answer>" in p:
                            p = p.split("<answer>")[1]
                        else:
                            p = "E"
                        pred.append(clean(p))
                        gt.append(answer)
                        # print(f"gt:{gt}, pred: {pred}")
                        if gt[-1] in pred[-1]:
                            cnt += 1
                    pbar.update(1)
            with open("output/pred.txt", "w") as f:
                f.write("\n".join([p for p in pred]))
            with open("output/gt.txt", "w") as f:
                f.write("\n".join([str(i) for i in gt]))
            print(len(pred) == len(gt))
            print(f"The accuracy of validation set is: {cnt/len(gt)*100}%")
            # print(f"The accuracy of validation set is: {accuracy_score(gt, pred) * 100}%")
        else:
            inputs = tokenizer("Tom is", return_tensors='pt')
            inputs = {key: value.to(device) for key, value in inputs.items()}
            print("Input: Tom is\nOutput:")
            model.to(device)
            model.eval()
            generated = model.generate(**inputs, repetition_penalty=1.2, max_new_tokens=100)[0]
            print(tokenizer.decode(generated.cpu().numpy(), skip_special_tokens=True))    
    
