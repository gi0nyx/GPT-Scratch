import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Embedding(nn.Module):
  def __init__(self, vocab_size, emb_dim):
    super().__init__()
    self.emb_mat = nn.Parameter(torch.randn(vocab_size,emb_dim, requires_grad=True))

  def forward(self,idx):

    return self.emb_mat[idx]

class PositionalEncoding(nn.Module):
  def __init__(self,seq_len,emb_dim):
    super().__init__()

    self.seq_len = torch.tensor(seq_len,requires_grad=False)
    self.emb_dim = torch.tensor(emb_dim,requires_grad=False)
    self.i = emb_dim//2
    self.pos_emb_mat = torch.zeros((self.seq_len,self.emb_dim),requires_grad = False)
    
    for t in range(self.seq_len):
      for i in range(self.i):
        power_term = 2*i/self.emb_dim
        self.pos_emb_mat[t,2*i] = torch.sin(t/(10000**power_term))
        self.pos_emb_mat[t,2*i+1] = torch.cos(t/(10000**power_term))
    
  def forward(self):

    return self.pos_emb_mat


class SelfAttention(nn.Module):
  def __init__(self,dim):
    super().__init__()
    self.dim = dim
    self.Q = nn.Linear(CFG.emb_dim,dim,bias=False)
    self.K = nn.Linear(CFG.emb_dim,dim,bias=False)
    self.V = nn.Linear(CFG.emb_dim,dim,bias=False)
    self.dropout = nn.Dropout(CFG.dropout)
    self.mask = torch.tril(torch.ones(CFG.seq_len, CFG.seq_len,device=CFG.device))


  def forward(self,seq):
    B,T,C = seq.shape
    q = self.Q(seq)  # seq (B,T,dim)
    k = self.K(seq)  # seq (B,T,dim)
    v = self.V(seq) # seq (B,T,dim)
    


    att_score = q@k.transpose(2,1)*self.dim**-0.5 # (B,T,dim) x (B,C,dim) -> (B,T,T)

    att_score = att_score.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

    att_score = F.softmax(att_score,dim=-1)
    att_score = self.dropout(att_score)

    att_score = att_score@v #(B,T,dim)

    return att_score


class MultiHead(nn.Module):
  def __init__(self,n_heads,dim):
    super().__init__()
    self.n_heads = n_heads
    self.dim = dim
    self.dropout = nn.Dropout(CFG.dropout)
    self.proj = nn.Linear(CFG.emb_dim, CFG.emb_dim)
    self.heads = nn.ModuleList([SelfAttention(self.dim) for _ in range(n_heads)])

  def forward(self,seq):
    out = torch.cat([head(seq) for head in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out


class FeedFoward(nn.Module):
  def __init__(self, att_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(att_dim, 4 * att_dim),
            nn.ReLU(),
            nn.Linear(4 * att_dim, att_dim),
            nn.Dropout(CFG.dropout),
        )
  def forward(self, x):
    return self.net(x)

class DecoderBlock(nn.Module):
  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = CFG.emb_dim // n_head
    self.att = MultiHead(n_head, head_size)
    self.ffwd = FeedFoward(CFG.emb_dim)
    self.ln1 = nn.LayerNorm(CFG.emb_dim)
    self.ln2 = nn.LayerNorm(CFG.emb_dim)

  def forward(self,x):
    x =  x + self.att(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x


class GPT(nn.Module):
  def __init__(self):
    super().__init__()

    self.emb = Embedding(CFG.vocab_size,CFG.emb_dim)
    self.pos = PositionalEncoding(CFG.seq_len,CFG.emb_dim)

    self.blocks = nn.Sequential(*[DecoderBlock(CFG.emb_dim, n_head=CFG.n_heads) for _ in range(CFG.n_layer)])
    self.ln_f = nn.LayerNorm(CFG.emb_dim)
    self.head = nn.Linear(CFG.emb_dim,CFG.vocab_size)

  def forward(self,x,y=None):

    B,T = x.shape

    x = self.emb(x)
    pos_enc = self.pos()[torch.arange(T)].to(CFG.device)

    x = x + pos_enc

    x = self.blocks(x)

    x = self.ln_f(x)

    x = self.head(x)

    if y != None:
      B,T,C = x.shape
      x = x.view(B*T,C)
      y = y.view(B*T)
      loss = F.cross_entropy(x,y)
    else:
      loss = None

    return x,loss

  
  def generate(self,idx,max_len_token=512):
    idx = idx.to(CFG.device)
    with torch.no_grad():
      for iter in range(max_len_token):

        pred = idx[:, -CFG.seq_len:]
        pred = pred.to(CFG.device)
        preds,loss = self(pred)

        preds = preds[:, -1, :]
        preds = F.softmax(preds,dim=-1)
        idx_next = torch.multinomial(preds, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx