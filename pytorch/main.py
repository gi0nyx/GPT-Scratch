
from token2 import BasicTokenizer
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class CFG:
  emb_dim = 64
  vocab_size = 256
  seq_len = 32
  att_dim = 64
  batch_size=16
  lr=1e-4
  iter = 30000
  eval_iters = 200
  dropout=0.0
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  n_heads = 4
  n_layer = 4


with open('twitter (1).txt', 'r', encoding='utf-8') as f:
    text = f.read()

token = BasicTokenizer()

data = torch.tensor(token.encode(text))

n = int(0.9*len(data))
train = data[:n]
val = data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train if split == 'train' else val
    ix = torch.randint(len(data) - CFG.seq_len, (CFG.seq_len,))
    x = torch.stack([data[i:i+CFG.seq_len] for i in ix])
    y = torch.stack([data[i+1:i+CFG.seq_len+1] for i in ix])
    x, y = x.to(CFG.device), y.to(CFG.device)
    return x, y

gpt = GPT().to(CFG.device)

opt = torch.optim.Adam(gpt.parameters(),lr=CFG.lr)

@torch.no_grad()
def estimate_loss():
    out = {}
    gpt.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(CFG.eval_iters)
        for k in range(CFG.eval_iters):
            X, Y = get_batch(split)
            X = X.to(CFG.device)
            Y = Y.to(CFG.device)
            logits, loss = gpt(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    gpt.train()
    return out

for iter in tqdm(range(CFG.iter)):
  x,y = get_batch('train')
  x = x.to(CFG.device)
  y = y.to(CFG.device)

  preds, loss = gpt.forward(x,y)

  opt.zero_grad()

  loss.backward()

  opt.step()

  if iter%300==0:
    print(estimate_loss())

context = torch.zeros((1, 1), dtype=torch.long, device=CFG.device)
print(token.decode(gpt.generate(context, 10000)[0].tolist()))

txt = token.decode(gpt.generate(context, 10000)[0].tolist())

with open('generated.txt', 'w') as file:
    file.write(txt)

