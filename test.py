import config as args
import  torch
from utils import get_model
tokenizer, model = get_model(args)
device = torch.device("cuda")
model.to(device)
inputs = tokenizer("Tom is", return_tensors='pt')
inputs = {key: value.to(device) for key, value in inputs.items()}
out = model(**inputs)
print(out.logits.shape)
res = out.logits.argmax(-1)
print(tokenizer.decode(res[0]))
# generated = model.generate(**inputs, repetition_penalty=1.2, max_new_tokens=100)[0]