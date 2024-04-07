import jax.numpy as jnp
from flax import linen as nn
import numpy as np

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
  n_heads = 4
  n_layer = 4
  max_len_token = 50

class PositionalEncoding(nn.Module):
  seq_len: int
  emb_dim: int

  def setup(self):
    self.i = self.emb_dim//2
    self.pos_emb_mat = np.zeros((self.seq_len,self.emb_dim))

    for t in range(self.seq_len):
      for i in range(self.i):
        power_term = 2*i/self.emb_dim
        self.pos_emb_mat[t,2*i] = np.sin(t/(10000**power_term))
        self.pos_emb_mat[t,2*i+1] = np.cos(t/(10000**power_term))

  @nn.compact
  def __call__(self):
    return jnp.array(self.pos_emb_mat)


class SelfAttention(nn.Module):
  dim: int
  def setup(self):

    self.Q = nn.Dense(self.dim,kernel_init=nn.initializers.xavier_uniform(),bias_init=nn.initializers.zeros)
    self.K = nn.Dense(self.dim,kernel_init=nn.initializers.xavier_uniform(),bias_init=nn.initializers.zeros)
    self.V = nn.Dense(self.dim,kernel_init=nn.initializers.xavier_uniform(),bias_init=nn.initializers.zeros)
    self.dropout = nn.Dropout(CFG.dropout,deterministic=False)
    self.mask = np.tril(np.ones((CFG.seq_len, CFG.seq_len)))


  def __call__(self,seq):
    B,T,C = seq.shape
    q = self.Q(seq)  # seq (B,T,dim)
    k = self.K(seq)  # seq (B,T,dim)
    v = self.V(seq) # seq (B,T,dim)


    att_score = jnp.matmul(q, jnp.transpose(k, (0, 2, 1)) * self.dim ** -0.5) # (B, T, dim) @ (B, dim, T) -> (B, T, T)

    att_score = jnp.where(self.mask[:T, :T] == 0, float('-inf'),att_score)

    att_score = nn.softmax(att_score,axis=-1)
    att_score = self.dropout(att_score)

    att_score = att_score@v #(B,T,dim)

    return att_score

class MultiHead(nn.Module):
  n_heads: int
  dim: int
  def setup(self):
    self.dropout = nn.Dropout(CFG.dropout,deterministic=False)
    self.proj = nn.Dense(CFG.emb_dim,kernel_init=nn.initializers.xavier_uniform(),bias_init=nn.initializers.zeros)
    self.heads = [SelfAttention(self.dim) for _ in range(self.n_heads)]

  def __call__(self,seq):
    out = jnp.concatenate([head(seq) for head in self.heads], axis=-1)
    out = self.dropout(self.proj(out))
    return out


class FeedFoward(nn.Module):
  att_dim: int
  def setup(self):
        self.net = nn.Sequential([
            nn.Dense(4 * self.att_dim,kernel_init=nn.initializers.xavier_uniform(),bias_init=nn.initializers.zeros),
            nn.relu,
            nn.Dense(self.att_dim,kernel_init=nn.initializers.xavier_uniform(),bias_init=nn.initializers.zeros),
            nn.Dropout(CFG.dropout,deterministic=False)]
        )
  def __call__(self, x):
    return self.net(x)

class DecoderBlock(nn.Module):
  n_embd:int
  n_head:int
  def setup(self):
    super().__init__()
    head_size = CFG.emb_dim // self.n_head
    self.att = MultiHead(self.n_head, head_size)
    self.ffwd = FeedFoward(CFG.emb_dim)
    self.ln1 = nn.LayerNorm()#(CFG.emb_dim)
    self.ln2 = nn.LayerNorm()#(CFG.emb_dim)

  def __call__(self,x):
    x =  x + self.att(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x


class GPT(nn.Module):
  def setup(self):
    self.emb = nn.Embed(CFG.vocab_size, CFG.emb_dim,embedding_init=nn.initializers.xavier_uniform())
    self.pos = PositionalEncoding(CFG.seq_len,CFG.emb_dim)

    self.blocks = nn.Sequential([DecoderBlock(CFG.emb_dim, CFG.n_heads) for _ in range(CFG.n_layer)])
    self.ln_f = nn.LayerNorm()
    self.head = nn.Dense(CFG.vocab_size,kernel_init=nn.initializers.xavier_uniform(),bias_init=nn.initializers.zeros)

  def __call__(self,x):

    B,T = x.shape

    x = self.emb(x)
    pos_enc = self.pos()[jnp.arange(T)]

    x = x + pos_enc

    x = self.blocks(x)

    x = self.ln_f(x)

    x = self.head(x)

    return x



