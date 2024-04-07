from tqdm import tqdm
import regex as re

class BasicTokenizer:
  '''
  uses bpe algorithm for creating subword tokens

  '''
  
  def __init__(self) -> None:
    self.vocab = [i for i in range(256)]

  def train(self,text,vocab_size,verbose=False):
   
    self.merges = dict()
    self.text = list(text.encode('utf-8'))
   
    self.vocab_size = vocab_size
    real_max_value_n = 0
    if self.vocab_size >= 256:
      self.epochs = self.vocab_size - 256
    for epoch in tqdm(range(self.epochs)):
      merge_value = 255+epoch+1
        

      print(self.text)
      res, max_value = self.most_freq_pair(self.text)

      self.text,self.vocab = self.merge(self.text,max_value,self.vocab,merge_value)

      if res[max_value] > real_max_value_n:
        real_max_value = max_value
        real_max_value_n = res[max_value]

      real_max_value_n = 0

      self.merges[merge_value] = real_max_value


    print('training completed')

  def encode(self,text):
    text = list(text.encode('utf-8'))
    merges_n = self.vocab[256:]
    new_text = []
    idx = 0
    for key in merges_n:
      while idx < len(text):
        if (self.merges[key][0] == text[idx]) and (self.merges[key][1] == text[idx+1]):
          new_text.append(key)

          idx = idx + 2
        else:

          new_text.append(text[idx])

          idx = idx + 1
      text = new_text
      new_text = []
      idx = 0

    return text


  def decode(self,ids):
    merges_n = self.vocab[256:]
    new_ids = []
    merges_n.reverse()
    for key in merges_n:

      for id in ids:
        if key == id:
          new_ids.append(self.merges[key][0])
          new_ids.append(self.merges[key][1])
        else:
          new_ids.append(id)

      ids = new_ids
      new_ids = []

    text_decode = b''
    text_decode= text_decode.join([bytes([idx]) for idx in ids])
    text = text_decode.decode("utf-8",errors="replace")

    return text



  def most_freq_pair(self,text):
    res = dict()
    pairs = [[text[i],text[i+1]] for i in range(len(text)-1)]
    for pair in pairs:
      pair_key = tuple(pair)
      if pair_key not in res.keys():
        res[pair_key] = 1
      else:
        res[pair_key] +=1


    max_value = max(res, key = lambda x: res[x])

    return res, max_value

  def merge(self,text,max_value,vocab,merge_value):
    ids = text
    i = 0
    new_ids = []

    while i < len(ids)-1:
      if ids[i] == max_value[0] and ids[i+1] == max_value[1]:
        if merge_value not in vocab:
          vocab.append(merge_value)

        new_ids.append(merge_value)
        i = i + 2
      else:
        new_ids.append(ids[i])
        i = i+1

    return new_ids,vocab




class GPTTokenizer:
  '''
  uses regex pattern 
  '''
  def __init__(self) -> None:
    self.vocab = [i for i in range(256)]

  def train(self,text,vocab_size,verbose=False):
    self.merges = dict()
    self.GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    gpt2pat = re.compile(self.GPT4_SPLIT_PATTERN)
    self.text = re.findall(gpt2pat,text)
    self.text = [list(text.encode('utf-8')) for text in self.text]
    self.vocab_size = vocab_size
    real_max_value_n = 0
    if self.vocab_size >= 256:
      self.epochs = self.vocab_size - 256
    for epoch in tqdm(range(self.epochs)):
      merge_value = 255+epoch+1
      for iter,item in enumerate(self.text):
        if len(self.text[iter]) == 1:
          continue



        res, max_value = self.most_freq_pair(item)

        item,self.vocab = self.merge(item,max_value,self.vocab,merge_value)

        if res[max_value] > real_max_value_n:
          real_max_value = max_value
          real_max_value_n = res[max_value]

        self.text[iter] = item
      real_max_value_n = 0

      self.merges[merge_value] = real_max_value

    print(self.merges)
    print('training completed')

  def encode(self,text):
    text = list(text.encode('utf-8'))
    merges_n = self.vocab[256:]
    new_text = []
    idx = 0
    for key in merges_n:
      while idx < len(text):
        if (self.merges[key][0] == text[idx]) and (self.merges[key][1] == text[idx+1]):
          new_text.append(key)

          idx = idx + 2
        else:

          new_text.append(text[idx])

          idx = idx + 1
      text = new_text
      new_text = []
      idx = 0

    return text


  def decode(self,ids):
    merges_n = self.vocab[256:]
    new_ids = []
    merges_n.reverse()
    for key in merges_n:

      for id in ids:
        if key == id:
          new_ids.append(self.merges[key][0])
          new_ids.append(self.merges[key][1])
        else:
          new_ids.append(id)

      ids = new_ids
      new_ids = []

    text_decode = b''
    text_decode= text_decode.join([bytes([idx]) for idx in ids])
    text = text_decode.decode("utf-8",errors="replace")

    return text



  def most_freq_pair(self,text):
    res = dict()
    pairs = [[text[i],text[i+1]] for i in range(len(text)-1)]
    for pair in pairs:
      pair_key = tuple(pair)
      if pair_key not in res.keys():
        res[pair_key] = 1
      else:
        res[pair_key] +=1


    max_value = max(res, key = lambda x: res[x])

    return res, max_value

  def merge(self,text,max_value,vocab,merge_value):
    ids = text
    i = 0
    new_ids = []

    while i < len(ids)-1:
      if ids[i] == max_value[0] and ids[i+1] == max_value[1]:
        if merge_value not in vocab:
          vocab.append(merge_value)

        new_ids.append(merge_value)
        i = i + 2
      else:
        new_ids.append(ids[i])
        i = i+1

    return new_ids,vocab