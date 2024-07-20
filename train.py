from transformerlib import TrainerCallback, AutoTokenizer, set_seed, AdamW, DataCollatorForLanguageModeling, Trainer, TrainingArguments, TrainerState
from torch.utils.data import DataLoader
# from datasets.iterable_dataset import IterableDataset
from datasets import load_dataset, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.nn import DataParallel
from torch.cuda import device_count
from collections import OrderedDict
# from accelerate import Accelerator
from numpy import mean, exp
from torch import cuda
from utils import *

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
        with open(f"./logs/log_{formatted_date}", "a") as f:
            f.write(str(logs) + "\n")
        

class CustomTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir

        self.model.lm_head.weight = torch.nn.Parameter(self.model.transformer.wte.weight.clone())
        self.model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        # super().save_model(f"{output_dir}/{self.model.__class__.__name__}.param", _internal_call=_internal_call)
        self.model.lm_head.weight = self.model.transformer.wte.weight


def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=args.max_input_length)


def sliding_window(text, max_length=1024, overlap=50):
    tokens = tokenizer.encode(text['text'])
    stride = max_length - overlap
    token_blocks = [tokens[i:i+max_length] for i in range(0, len(tokens), stride) if i+max_length <= len(tokens)]
    return token_blocks

    
def prepare_eval(samples, batch_size):
    res = []
    ans = []
    choices = ["A", "B", "C", "D"]
    batch_index = 0
    for question, c, answer in zip(samples["question"], samples["choices"], samples["answer"]):
        choice = "; ".join(f"{i}. {j}" for i, j in zip(choices, c))
        sample = f"<question>{question}</question><choice>{choice}</choice><answer>"
        res.append(sample)
        ans.append(answer)
    
    while batch_index < len(res):
        yield tokenizer(res[batch_index:batch_index+batch_size], return_tensors='pt', truncation=True, \
            padding='max_length', max_length=args.max_input_length), ans[batch_index:batch_index+batch_size]
        batch_index += batch_size


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


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(args.model_name) #, cache_dir=cache_dir)
    model = get_model(args)
    tokenizer.pad_token = tokenizer.eos_token
    set_seed(37)

    #############################
    if args.tuning_mode:
        dataset = load_dataset(args.tuning_set, "all")
        # split the dataset into train and validation
        # train_idx = int(len(dataset) * 0.9)
        train_dataset = dataset["auxiliary_train"]  # [:train_idx]  auxiliary_train
        valid_dataset = dataset["validation"]  # [train_idx:]
        # train_dataset = Dataset.from_dict(train_dataset)
        # valid_dataset = Dataset.from_dict(valid_dataset)
        tokenized_train_dataset = train_dataset.map(instruct, batched=True)
        tokenized_valid_dataset = valid_dataset.map(instruct, batched=True)
    else:
        dataset = load_dataset(args.dataset, streaming=True)
        train_dataset = dataset["train"] #load_dataset(args.dataset, split="train", streaming=True)
        valid_dataset = dataset["validation"] #load_dataset(args.dataset, split="validation", streaming=True)
        valid_dataset = Dataset.from_dict({"text": get_subset(valid_dataset, args.eval_sample)})
        tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
        tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
                    output_dir='./results',          # 输出目录
                    evaluation_strategy="steps",
                    lr_scheduler_type="cosine",
                    eval_steps=args.eval_steps,
                    overwrite_output_dir=True,
                    num_train_epochs=args.epochs,              # 训练轮数
                    per_device_train_batch_size=args.batch_size,   # 每个设备的批次大小
                    per_device_eval_batch_size=args.batch_size * 4,
                    save_total_limit=2,              # 保存模型的总数限制
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
    
    trainer = CustomTrainer(
                        model=model,
                        args=training_args,
                        data_collator=data_collator,
                        train_dataset=tokenized_train_dataset,
                        eval_dataset=tokenized_valid_dataset,
                        callbacks=[CustomCallback()]
                    )
    
    if args.checkpoint_dir:
        trainer_state = TrainerState.load_from_json(os.path.join(args.checkpoint_dir, 'trainer_state.json'))
        trainer.state = trainer_state
    
    try:
        trainer.train()
    except Exception as e:
        print("Error: %s"%e)
        trainer.save_model()
    trainer.evaluate()
    ##########################################################
    
    # ----------------------- unit test ------------------------ #
    # dataset = DecodersDataset(texts, tokenizer, max_length=512, test=True)
    # train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # val_dataloader = None
    # ---------------------------------------------------------- #
    
    del trainer
    del tokenized_train_dataset
    del train_dataset
    cuda.empty_cache()
    
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
            for samples, answers in prepare_eval(valid_dataset, args.batch_size*6):
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
                    if choices[gt[-1]] in pred[-1]:
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
    
