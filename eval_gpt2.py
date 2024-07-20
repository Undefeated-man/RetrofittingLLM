from transformerlib import TrainerCallback, AutoTokenizer, set_seed, AdamW, DataCollatorForLanguageModeling, Trainer, TrainingArguments, TrainerState
# from datasets.iterable_dataset import IterableDataset
from datasets import load_dataset, Dataset
# from accelerate import Accelerator
from numpy import exp
from torch import cuda
from utils import *
from dataclasses import dataclass

import config as args
import datetime
import tqdm
import torch
import os
import warnings
warnings.filterwarnings("ignore")
    
@dataclass
class PromptForQA:
    who = "answer with a name from the document in response to a who question"
    when = "answer with a date or a time from the document in response to a when question"
    tf = "answer with a yes or no from the document in response to a true/false question"
    

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


def find_prompt(text):
    tf_questions = ["Is", "Are", "Was", "Were", "Did", "Do", "Does", "Can", "Could", "Will", "Would", "Should", "Has", "Have", "Had"]
    if text.startswith("Who"):
        return f"Hint: {PromptForQA.who}"
    elif text.startswith("When"):
        return f"Hint: {PromptForQA.when}"
    elif text.split()[0] in tf_questions:
        return f"Hint: {PromptForQA.tf}"
    return ""


def prepare_eval(samples, batch_size, q_type="MCQ"):
    res = []
    ans = []
    batch_index = 0
    if q_type == "MCQ":
        choices = ["A", "B", "C", "D"]
        for question, c, answer in zip(samples["question"], samples["choices"], samples["answer"]):
            choice = "; ".join(f"{i}. {j}" for i, j in zip(choices, c))
            sample = f"<question>{question}</question><choice>{choice}</choice><answer>"
            res.append(sample)
            ans.append(answer)
    elif q_type == "TF":
        for question, answer in zip(samples["question"], samples["answer"]):
            sample = f"<question>{question}</question><answer>{answer}</answer>"
            res.append(sample)
            ans.append(answer)
    elif q_type == "QA":
        with tqdm.tqdm(total=len(samples)) as pbar:
            pbar.set_description("Preparing samples")
            for questions, context, answers in zip(samples["questions"], samples["story"], samples["answers"]):
                for question, answer in zip(questions, answers["input_text"]):
                    sample = f"Question: {question} Context: {context} Answer:"
                    if len(sample) < 950:
                        sample = f"{find_prompt(question)} {sample}"
                    elif len(sample) > 1020:
                        continue
                    res.append(sample)
                    ans.append(answer)
                pbar.update(1)
    
    while batch_index < len(res):
        yield tokenizer(res[batch_index:batch_index+batch_size], return_tensors='pt', truncation=True, \
            padding='max_length', max_length=args.max_input_length), ans[batch_index:batch_index+batch_size]
        batch_index += batch_size


def instruct(samples, q_type="MCQ"):
    """
        Instruct the model to generate the text
        
        Args:
            samples: list, the samples
            q_type: str, the type of question. Default is "MCQ", choices are "MCQ", "TF" and "QA"
        
        Returns:
            list, the generated text
    """
    res = []
    if q_type == "MCQ":
        choices = ["A", "B", "C", "D"]
        for question, c, answer in zip(samples["question"], samples["choices"], samples["answer"]):
            choice = "; ".join(f"{i}. {j}" for i, j in zip(choices, c))
            sample = f"<question>{question}</question><choice>{choice}</choice><answer>{choices[answer]}</answer>"
            res.append(sample)
    elif q_type == "TF":
        for question, answer in zip(samples["question"], samples["answer"]):
            sample = f"<question>{question}</question><answer>{answer}</answer>"
            res.append(sample)
    elif q_type == "QA":
        for questions, context, answers in zip(samples["questions"], samples["story"], samples["answers"]):
            for question, answer in zip(questions, answers):
                sample = f"Question: {question} Context: {context} Answer: {answer}"
                res.append(sample)
    return tokenizer(res, return_tensors='pt', truncation=True, \
        padding='max_length', max_length=args.max_input_length)


def get_subset(iterable_dataset, num_samples):
    subset_data = []
    for i, sample in enumerate(iterable_dataset):
        if i >= num_samples:
            break
        subset_data.append(sample["text"])
    return subset_data


def clean(text, q_type="MCQ"):
    if q_type == "MCQ":
        text = text.replace("</answer>", "")
        text = text.replace("</", "")
    elif q_type == "TF":
        text = text.replace("</answer>", "")
        text = text.replace("</", "")
    elif q_type == "QA":
        text = text.replace("\n", " ")
    return text.strip()


def find_longest_ans(ds):
    df = ds["validation"].to_pandas()
    df["longest_ans"] = df["answers"].apply(lambda x: max([len(i.split()) for i in x["input_text"]]))
    return df["longest_ans"].max()
    

def generate(model, input_ids, max_new_tokens=30, past_key_values=None):
    res = []
    for _ in range(max_new_tokens):
        outputs = model(input_ids=input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits
        next_token_logits = logits[:, -1, :].argmax(dim=-1)
        input_ids = torch.cat([input_ids, next_token_logits.unsqueeze(-1)], dim=1)
        # res.append(int(input_ids.cpu()))
        past_key_values = outputs.past_key_values
        if next_token_logits.item() == tokenizer.eos_token_id:
            break
    return input_ids.cpu().numpy(), past_key_values
    


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(args.model_name) #, cache_dir=cache_dir)
    model = get_model(args)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    set_seed(37)

    #############################
    valid_dataset = load_dataset(args.eval_dataset, split="validation")  # [train_idx:]

    ##########################################################
    
    # Output an example to see the result
    device = torch.device("cuda:0")
    q_type = "QA"
    model.to(device)
    model.eval()
    
    if q_type == "MCQ":
        # from sklearn.metrics import accuracy_score
        pred = []
        gt = []
        cnt = 0
        choices = ["A", "B", "C", "D"]
        with tqdm.tqdm(total=round(len(valid_dataset)/args.eval_batch_size)) as pbar:
            for samples, answers in prepare_eval(valid_dataset, args.eval_batch_size):
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
    elif q_type == "TF":
        raise NotImplementedError
    elif q_type == "QA":
        pred = []
        gt = []
        cnt = 0
        questions = valid_dataset["questions"]
        answers = valid_dataset["answers"]
        stories = valid_dataset["story"]
        with tqdm.tqdm(total=len(valid_dataset)) as pbar:
            for i in range(len(stories)):
                pbar.update(1)
                past_key_values = None
                sample = f"Context: {stories[i]}"  " Question: {questions[i][0]} Answer:"
                
                for q, a in zip(questions[i], answers[i]["input_text"]):
                    if past_key_values is None:
                        sample = f"Context: {stories[i]} Question: {q} Answer:"
                    else:
                        sample = f"Question: {q} Answer:"
                    
                    if len(sample) < 950:
                        sample = f"{find_prompt(q)} {sample}"
                    elif len(sample) > 1020:
                        continue
                    
                    print("**"*50)
                    input_ = tokenizer(sample, return_tensors='pt', truncation=True, max_length=args.max_input_length).to(device)
                    # input_ = {key: value for key, value in input_.items()}
                    output, past_key_values = generate(model, input_ids=input_["input_ids"], past_key_values=past_key_values)
                    prediction = tokenizer.decode(output[0], skip_special_tokens=True)
                    print(f"Prediction: {prediction}")
                    input()
                    pred.append(prediction)
                    gt.append(a)
            
            
        # for samples, answers in prepare_eval(valid_dataset, args.eval_batch_size, q_type="QA"):
        #     cnt += 1
        #     dots = ".."*(cnt%3)
        #     print(f"{cnt} Evaluating.{dots}                 ")
        #     # if cnt < 547:
        #     #     continue
        #     # print(tokenizer.eos_token, tokenizer.eos_token_id)
        #     # input()
        #     # with open("output/sample.txt", "w") as f:
        #     #     f.write(str(samples["input_ids"][0].tolist()))
        #     past_key_values = None
        #     samples = {key: value.to(device) for key, value in samples.items()}
        #     generated = model.generate(**samples, max_new_tokens=30, past_key_values=past_key_values, top_k=50, top_p=0.95, temperature=0.7)
        #     for sample, answer in zip(generated, answers):
        #         p = tokenizer.decode(sample.cpu().numpy(), skip_special_tokens=True)
        #         pred.append(clean(p, "QA"))
        #         gt.append(answer)
                
        dataset_name = args.eval_dataset.split("/")[-1]
        with open(f"output/{dataset_name}_pred.txt", "w") as f:
            f.write("\n\n".join([p for p in pred]))
        with open(f"output/{dataset_name}_gt.txt", "w") as f:
            f.write("\n".join([str(i) for i in gt]))
        print("\n\n")
        
    else:
        inputs = tokenizer("Tom is", return_tensors='pt')
        inputs = {key: value.to(device) for key, value in inputs.items()}
        print("Input: Tom is\nOutput:")
        model.to(device)
        model.eval()
        generated = model.generate(**inputs, repetition_penalty=1.2, max_new_tokens=50)[0]
        print(tokenizer.decode(generated.cpu().numpy(), skip_special_tokens=True))    
    
