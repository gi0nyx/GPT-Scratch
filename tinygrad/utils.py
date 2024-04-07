import numpy as np
from tinygrad.helpers import Timing
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from tinygrad import nn, Device
from tinygrad import dtypes
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import fetch
from tinygrad.helpers import getenv, colored


class CFG:
  emb_dim = 64
  vocab_size = 256
  seq_len = 32
  att_dim = 64
  batch_size=32
  lr=1e-4
  iter = 10000
  eval_iters = 300
  dropout=0.0
  n_heads = 4
  n_layer = 4

class PositionalEncoding():
  def __init__(self,seq_len,emb_dim):
    
    self.seq_len = seq_len
    self.emb_dim = emb_dim
    self.i = emb_dim//2
    self.pos_emb_mat = np.zeros((self.seq_len,self.emb_dim))
    
    for t in range(self.seq_len):
      for i in range(self.i):
        power_term = 2*i/self.emb_dim
        self.pos_emb_mat[t,2*i] = np.sin(t/(10000**power_term))
        self.pos_emb_mat[t,2*i+1] = np.cos(t/(10000**power_term))
    
  def __call__(self):

    return Tensor(self.pos_emb_mat,dtype=dtypes.float,requires_grad=False)


class SelfAttention():
  def __init__(self,emb_dim,dim):
    self.dim = dim
    self.Q = nn.Linear(emb_dim,dim,bias=False)
    self.K = nn.Linear(emb_dim,dim,bias=False)
    self.V = nn.Linear(emb_dim,dim,bias=False)
    self.mask = Tensor.tril(Tensor.ones(CFG.seq_len, CFG.seq_len))


  def __call__(self,seq):
    B,T,C = seq.shape
    q = self.Q(seq)  # seq (B,T,dim)
    k = self.K(seq)  # seq (B,T,dim)
    v = self.V(seq) # seq (B,T,dim)

    att_score = q.scaled_dot_product_attention(k,v,attn_mask=self.mask[:T, :T],dropout_p=CFG.dropout)

    return att_score


class MultiHead():
  def __init__(self,n_heads,dim,emb_dim):
    self.n_heads = n_heads
    self.dim = dim
    self.dropout = Tensor.dropout
    self.proj = nn.Linear(emb_dim, emb_dim)
    self.heads = [SelfAttention(emb_dim,self.dim) for _ in range(n_heads)]

  def __call__(self,seq):
    seq = seq.sequential(self.heads)
    
    #seq = seq.reshape(-1,seq.shape[1],seq.shape[2]*seq.shape[3])
    seq = self.dropout(self.proj(seq),CFG.dropout)
    return seq


class FeedFoward():
  def __init__(self, att_dim):
    self.l1 = nn.Linear(att_dim, 4 * att_dim)
    self.l2 = nn.Linear(4 * att_dim, att_dim)
        
  def __call__(self, x):
    x = self.l1(x)
    x = Tensor.relu(x)
    x = self.l2(x)
    x = Tensor.dropout(x,CFG.dropout)
    return x

class DecoderBlock():
  def __init__(self, n_embd, n_head):
    head_size = CFG.emb_dim // n_head
    self.att = MultiHead(n_head, CFG.emb_dim,n_embd)
    self.ffwd = FeedFoward(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def __call__(self,x):
    x =  x + self.att(x.layernorm())
    x = x + self.ffwd(self.ln2(x))
    return x

class GPT:
  def __init__(self):
    self.emb = nn.Embedding(CFG.vocab_size,CFG.emb_dim)
    self.ln_f = nn.LayerNorm(CFG.emb_dim)
    self.block = [DecoderBlock(CFG.emb_dim, n_head=CFG.n_heads) for _ in range(CFG.n_layer)]
    #self.l1 = nn.Linear(CFG.emb_dim,CFG.emb_dim)
    self.head = nn.Linear(CFG.emb_dim,CFG.vocab_size)
    self.pos = PositionalEncoding(CFG.seq_len,CFG.emb_dim)

  def __call__(self, x,y=None):
    B,T = x.shape
    x = self.emb(x)
    
    pos_enc = self.pos()[Tensor.arange(T)]
    x = x + pos_enc
    
    x = x.sequential(self.block)
   
    x = self.ln_f(x)
    x = self.head(x)
    
    if y is not None:
        B,T,C = x.shape
        x = x.reshape(B*T,C)
        y = y.reshape(B*T)
        return x,y
    return x


  def generate(self,idx,max_len_token=512):
        for iter in range(max_len_token):
            pred = idx[:, -CFG.seq_len:]
            preds = self(pred)
            preds = preds[:, -1, :]
            preds = preds.softmax()
            idx_next = Tensor.multinomial(preds,num_samples=1)
            idx = Tensor.cat(idx, idx_next, dim=1)
        return idx



