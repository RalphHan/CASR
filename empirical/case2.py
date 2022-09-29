import os
task=os.environ['TASK']
generation=os.environ['GENERATION']
ids={
    'kvret':19,
    'mtop':1362,
    'webqsp':167,
}
import torch
from transformers import T5Tokenizer
tk=T5Tokenizer.from_pretrained('t5-base')
print(tk.decode(torch.load(generation)[ids[task]],skip_special_tokens=True))