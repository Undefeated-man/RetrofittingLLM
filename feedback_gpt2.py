from transformers import TrainerCallback, AutoTokenizer, AutoConfig, GPT2Config, PretrainedConfig, set_seed, AdamW, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
from transformerlib.models.feedback_gpt2 import modeling_gpt2 as fbgpt2
from dataset import StreamDataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn import DataParallel
from torch.cuda import device_count
from collections import OrderedDict
from numpy import mean, exp

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
            logits: torch.Tensor, the logits from the model
            labels: torch.Tensor, the labels
    """
    return torch.exp(-torch.log(torch.gather(logits, 1, labels.unsqueeze(1))).mean())

class DecodersDataset(Dataset):
    """
    Return the dataset for the decoders training
    """
    def __init__(self, texts, tokenizer, max_length, test=False):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        self.test = test

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(self.texts[idx], 
                                 return_tensors='pt', 
                                 max_length=self.max_length, 
                                 truncation=True, 
                                 padding='max_length')
        
        tokens_len = torch.argwhere(encoded['input_ids'] == self.tokenizer.eos_token_id)
        if len(tokens_len) > 0:
            tokens_len = tokens_len[0][-1]
        else:
            tokens_len = self.max_length
        
        encoded['labels'] = encoded.input_ids.detach().clone()
        if self.test:  # output the test sample
            encoded['attention_mask'][0, tokens_len-2:] = 0
            encoded['input_ids'][0, tokens_len-2:] = self.tokenizer.eos_token_id
        return encoded

def train(model, dataloader, optimizer, test_dataloader=None, num_epochs=3, accumulation_steps=1, scaler=None):
    """
        Train the model
        
        Args:
            model: nn.Module, the model to train
            dataloader: DataLoader, the dataloader for the train dataset
            test_dataloader: DataLoader, the dataloader for the test dataset
            optimizer: torch.optim, the optimizer for the model
            num_epochs: int, the number of epochs
    """
    if device_count() > 1:
        model = DataParallel(model).cuda()
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
        if test_dataloader and epoch % 10 == 0:
            tmp_perx = evaluate(model, test_dataloader)
            if tmp_perx < best_perplexity:
                best_perplexity = tmp_perx
                stop_cnt = 0
                torch.save(
                    {
                        "feedback_gpt2":{
                            'epoch': epoch,
                            'model_state_dict': model.state_dict()
                        }
                    }, f"output/{model.__class__.__name__}.params"
                )
            else:
                stop_cnt += 1
            
def evaluate(model, dataloader):
    """
        Evaluate the model
        
        Args:
            model: nn.Module, the model to evaluate
            dataloader: DataLoader, the dataloader for the dataset
            
        Returns:
            float, the perplexity
    """
    model.eval()
    outputs_list = []
    labels_list = []
    for batch in dataloader:
        inputs = {key: value.to(model.device) for key, value in batch.items()}
        out = model(**inputs)
        outputs_list += out.logits
        labels_list += inputs['labels']
    perx = perplexity(outputs_list, labels_list)
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
        with open(f"./log_{formatted_date}", "a") as f:
            f.write(str(logs) + "\n")
        

class CustomTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir

        self.model.lm_head.weight = torch.nn.Parameter(self.model.transformer.wte.weight.clone())
        super().save_model(f"{output_dir}/{self.model.__class__.__name__}.param", _internal_call=_internal_call)
        self.model.lm_head.weight = self.model.transformer.wte.weight


if __name__ == "__main__":
    # set the cache dir
    debug = not True
    if debug:
        cache_dir = "/work/tc055/tc055/vincent/cache"
    else:
        cache_dir = "/home/s2497456/mnt/workdir/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = cache_dir

    tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
    config = AutoConfig.from_pretrained('gpt2', cache_dir=cache_dir)
    # config = GPT2Config(
    #     vocab_size=50257,
    #     n_positions=1024,
    #     n_ctx=1024,
    #     n_embd=768,
    #     n_layer=12,
    #     n_head=12,
    #     # Additional custom configurations can be added here
    # )
    print(isinstance(config, PretrainedConfig))
    print(type(config))
    model = fbgpt2.FBGPT2LMHeadModel(config=config)
    tokenizer.pad_token = tokenizer.eos_token
    set_seed(37)

    # initialize
    preload = False
    num_epochs = 5
    batch_size = 20
    if preload:
        prefetch_factor = 4  # preload the next n batches
    else:
        prefetch_factor = 0
    accumulation_steps = 10
    # texts = ["Hello, world!", "Hi, dear Tom!"]

    if debug:
        pth = "/work/tc055/tc055/vincent/RetrofittingLLM/dataset/SlimPajama"
    else:
        pth = "/home/s2497456/mnt/workdir/RetrofittingLLM/dataset/SlimPajama"

    max_length = 512
    scaler = GradScaler()
    # dataset = StreamDataLoader(pth, max_length=max_length)
    # dataset.stream_load()
    # dataset.test_load()
    # print(dataset.data["train"][:3])
    # print(type(dataset.data["train"]))
    # train_dataset = DecodersDataset(dataset.data["train"], tokenizer, max_length=max_length)
    # val_dataset = DecodersDataset(dataset.data["validation"], tokenizer, max_length=max_length)
    # test_dataset = DecodersDataset(dataset.data["test"], tokenizer, max_length=max_length)

    if preload:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,\
                                    prefetch_factor=prefetch_factor, pin_memory = True, persistent_workers=True)
        val_dataloader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2, \
                                    prefetch_factor=prefetch_factor, pin_memory = True, persistent_workers=True)
        test_dataloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, \
                                    prefetch_factor=prefetch_factor, pin_memory = True, persistent_workers=True)
    else:
        train_dataset = [TextDataset(
                          tokenizer=tokenizer,
                          file_path=f"dataset/processed/train_{i}.txt",  # 你的训练文件
                          block_size=max_length) for i in range(1, 234)]
        valid_dataset = TextDataset(
                          tokenizer=tokenizer,
                          file_path="dataset/processed/valid.txt",  # 你的验证文件
                          block_size=max_length)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
                      output_dir='./results',          # 输出目录
                      evaluation_strategy="steps",
                      eval_steps=20000,
                      overwrite_output_dir=True,
                      num_train_epochs=2,              # 训练轮数
                      per_device_train_batch_size=16,   # 每个设备的批次大小
                      save_steps=10_000,               # 保存模型的步数
                      save_total_limit=2,              # 保存模型的总数限制
                      logging_dir='./logs',
                      report_to=None
                    )
    for i in range(233):
        trainer = Trainer(
                      model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=train_dataset[i],
                      eval_dataset=valid_dataset,
                      callbacks=[CustomCallback()]
                    )
        trainer.train()
        # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # val_dataloader   = DataLoader(val_dataset, batch_size=batch_size * 3, shuffle=True)
        # test_dataloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # ----------------------- unit test ------------------------ #
    # dataset = DecodersDataset(texts, tokenizer, max_length=512, test=True)
    # train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # val_dataloader = None
    # ---------------------------------------------------------- #
    
    # optimizer = AdamW(model.parameters(), lr=5e-5)

    # training
    # train(model, train_dataloader, optimizer, val_dataloader, num_epochs=num_epochs)
    
    # testing
    # evaluate(model, test_dataloader)
        
    # Output an example to see the result
    inputs = tokenizer("Hello, ", return_tensors='pt')
    print(tokenizer.decode(model.generate(**inputs, max_new_tokens=4)[0]))    
    
    # dataset = DecodersDataset(texts, tokenizer, max_length=512, test=True)
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # model.eval()
    # for batch in dataloader:
    #     inputs = {key: value.to(model.device) for key, value in batch.items()}
    #     out = model(**inputs)
    #     print(out.logits.shape)
    #     print(torch.argmax(out.logits[0][0], dim=-1))
    #     print(tokenizer.decode(torch.argmax(out.logits[0][0], dim=-1)))

    #text = "Hello"
    #encoded_input = tokenizer(text, return_tensors='pt')
    #output = model.generate(**encoded_input, num_return_sequences=1)
    #print(tokenizer.decode(output[0], skip_special_tokens=True))

    #print(model("Hello, ", max_length=30, num_return_sequences=1))
