from token2 import BasicTokenizer
from utils import *
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from flax import linen as nn
import optax
import numpy as np
from tqdm import tqdm
import regex as re


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

def create_train_step(key, model, optimiser):
  params = model.init(key, jnp.zeros((CFG.batch_size,CFG.seq_len),dtype=jnp.int32))
  opt_state = optimiser.init(params)
  return params,opt_state

def loss_func(params,x,y):
  preds = gpt.apply(params,x)
  B,T,C = preds.shape
  preds = jnp.reshape(preds,(B*T,C))
  y = jnp.reshape(y,(B*T))
  loss = optax.softmax_cross_entropy_with_integer_labels(logits=preds,labels=y).mean()
  return loss

@jax.jit
def train_step(params,opt_state,x,y):
  losses, grads = jax.value_and_grad(loss_func)(params,x,y)
  updates, opt_state = optimiser.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state

def generate(params,idx,main_key):

  for iter in tqdm(range(CFG.max_len_token)):
    pred = idx[:, -CFG.seq_len:]

    preds = gpt.apply(params,pred)

    preds = preds[:, -1, :]
    #preds = nn.activation.softmax(preds)
    main_key, rng = jax.random.split(main_key)
    idx_next = jax.random.categorical(rng, preds,shape=(1,)) # applies softmax I think
    

    idx = jnp.hstack((idx, jnp.reshape(idx_next,(1,1))))

  return idx

@jax.jit
def v_generate(params,idx,main_key):
    return jax.vmap(generate,(None,0,0))(params,idx,main_key)



with open('twitter (1).txt', 'r', encoding='utf-8') as f:
    text = f.read()

token = BasicTokenizer()

data = jnp.array(token.encode(text))

n = int(0.9*len(data))
train = data[:n]
val = data[n:]

def get_batch(split,key):
    # generate a small batch of data of inputs x and targets y
    data = train if split == 'train' else val
    ix = np.random.randint(low=0,high=len(data) - CFG.seq_len, size=(CFG.seq_len,))
    x = jnp.stack([data[i:i+CFG.seq_len] for i in ix])
    y = jnp.stack([data[i+1:i+CFG.seq_len+1] for i in ix])
    return x, y


gpt = GPT()
key = jax.random.PRNGKey(0)
key, model_key = jax.random.split(key)
optimiser = optax.adam(learning_rate=CFG.lr)

params, opt_state = create_train_step(model_key, gpt, optimiser)

for iter in tqdm(range(CFG.iter)):

  key, subkey = jax.random.split(key)

  x,y = get_batch('train',subkey)
  x = jnp.array(x)
  y = jnp.array(y)

  #preds = gpt.apply(params,x)

  params,opt_state = train_step(params,opt_state,x,y)


  if iter%300==0:
    train_loss = loss_func(params,x,y)
    
    x,y = get_batch('valid',subkey)
    
    valid_loss = loss_func(params,x,y)
    
    print(f'train loss: {train_loss} valid loss: {valid_loss}')


context = jnp.zeros((20,1, 1), dtype=jnp.int32)
subkey = jax.random.split(key,20)
txt = v_generate(params,context,subkey)

def decode(txt):
    text = ''
    txt = [txt[i][0].tolist() for i in range(len(txt))]
    for item in txt:
        text += token.decode(item)
    return text

result = decode(txt)

print(result)
