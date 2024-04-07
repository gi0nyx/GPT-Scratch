
from token2 import BasicTokenizer
from utils import *
import numpy as np
from tinygrad.helpers import Timing
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from tinygrad import nn, Device
from tinygrad import dtypes
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import fetch
from tinygrad import TinyJit
from tinygrad.helpers import getenv, colored
from tqdm import tqdm
import regex as re

class CFG:
  emb_dim = 64
  vocab_size = 256
  seq_len = 32
  att_dim = 64
  batch_size=32
  lr=1e-4
  iter = 100
  eval_iters = 300
  dropout=0.0
  #device = 'cuda' if torch.cuda.is_available() else 'cpu'
  n_heads = 4
  n_layer = 4


with open('twitter (1).txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))

token = BasicTokenizer()

data = np.array(token.encode(text))

n = int(0.9*len(data)) # first 90% will be train, rest val
train = data[:n]
val = data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train if split == 'train' else val
    ix = np.random.randint(low=0,high=len(data) - CFG.seq_len, size=(CFG.batch_size,))
    x = np.stack([data[i:i+CFG.seq_len] for i in ix])
    y = np.stack([data[i+1:i+CFG.seq_len+1] for i in ix])
    return x, y

def sparse_categorical_crossentropy(self, Y, ignore_index=-1) -> Tensor:
    loss_mask = Y != ignore_index
    y_counter = Tensor.arange(self.shape[-1], dtype=dtypes.int32, requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), self.shape[-1])
    y = ((y_counter == Y.flatten().reshape(-1, 1)).where(-1.0, 0) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
    return self.log_softmax().mul(y).sum() / loss_mask.sum()


def estimate_loss():
    loss_dict = {}
    for split in ['train', 'val']:
        losses = np.zeros(10)
        for k in range(10):
            x, y = get_batch(split)

            x = Tensor(x, requires_grad=False)
            y = Tensor(y)

            out,y = net(x, y)
            loss = sparse_categorical_crossentropy(out, y).numpy()
            losses[k] = loss.item()
        loss_dict[split] = losses.mean()
    return loss_dict


net = GPT()

opt = Adam(get_parameters(net), lr=1e-4)

with Tensor.train():
  for step in tqdm(range(CFG.iter)):
    
    x, y = get_batch('train')
    x = Tensor(x, requires_grad=False)
    y = Tensor(y)
    

    
    out,y = net(x,y)
    
    
    loss = sparse_categorical_crossentropy(out, y)

    
    opt.zero_grad()

    
    loss.backward()

    
    opt.step()

    

    if step % CFG.eval_iters == 0:
      print(f"{estimate_loss()}")

context = Tensor.zeros((1, 1),requires_grad=False)
result = net.generate(context, 20)[0].numpy()
print(token.decode(result.astype(int).tolist()))
