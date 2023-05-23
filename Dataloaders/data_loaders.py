import random
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from torch.utils.data import DataLoader, BatchSampler, RandomSampler,SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import torch
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.ERROR)

import sys
sys.path.append('/content/drive/MyDrive/Master_thesis/Metrics_Evaluations')
import metrics_eval

#Data loaders file
'''
Data_laoder for GAN-BERT and FairGANBERT model
'''

class SS_GAN_Data_loader(Dataset):
  def __init__(self, data, tokenizer,max_len):
    print("data",data.columns)
    self.Text = data['Text']
    self.Label = data['target']
    self.identity_gender_terms = data['gender']
    self.identity_race_terms = data['race']
    self.label_mask = data['label_mask']
    self.index = data['index']
    self.attr = data['attr']

    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.Text)

  def __getitem__(self, item):
    Text = str(self.Text[item])
    target = self.Label[item]
    identity_gender_term = self.identity_gender_terms[item]
    identity_race_term = self.identity_race_terms[item]
    label_mask = self.label_mask[item]
    index  = self.index[item]
    attr  = self.attr[item]

    encoding = self.tokenizer.encode_plus(
      Text,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      truncation=True,
      return_tensors='pt',
    )
    return {
      'review_text': Text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long),
      'identity_gender_term': torch.tensor(identity_gender_term, dtype=torch.long),
      'identity_race_term':torch.tensor(identity_race_term, dtype=torch.long),
      'label_mask' : label_mask,
      'index':index,
      'attr':attr

    }

def ss_gan_data_loader(data,tokenizer, max_len, batch_size):
  dataset = SS_GAN_Data_loader(
    data,
    tokenizer,
    max_len
  )

  # subsample data to be mix of label and unlabel data
  size_label_data = data[data.target !=0].shape[0]

  try:
    unlabel = data[data.target ==0].sample(n =size_label_data )
    label = data.drop(index = data[data.target ==0].index)
    data1 = pd.concat([unlabel,label]).reset_index(drop=True)
    dataset = SS_GAN_Data_loader(
    data1,
    tokenizer,
    max_len
    )
  except:
    print("validation data")

  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
  ),dataset

def rnd_batch_ss_gan_data_loader(dataset, batch_size, num_batch =2):

  sampler = RandomSampler(dataset,False,batch_size*num_batch )

  return DataLoader(dataset, sampler=sampler, shuffle=False, batch_size=batch_size, num_workers=1)


class SS_IW_GAN_Data_loader(Dataset):

  def __init__(self, data, tokenizer,max_len):

    self.Text = data['Text']
    self.Label = data['target']
    self.identity_gender_terms = data['gender']
    self.identity_race_terms = data['race']
    self.label_mask = data['label_mask']
    self.index = data['index']
    self.attr = data['attr']
    self.weight = data['weights']


    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.Text)

  def __getitem__(self, item):
    Text = str(self.Text[item])
    target = self.Label[item]
    identity_gender_term = self.identity_gender_terms[item]
    identity_race_term = self.identity_race_terms[item]
    label_mask = self.label_mask[item]
    index  = self.index[item]
    attr  = self.attr[item]
    weight  = self.weight[item]


    encoding = self.tokenizer.encode_plus(
      Text,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      truncation=True,
      return_tensors='pt',
    )
    return {
      'review_text': Text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long),
      'identity_gender_term': torch.tensor(identity_gender_term, dtype=torch.long),
      'identity_race_term':torch.tensor(identity_race_term, dtype=torch.long),
      'label_mask' : label_mask,
      'index':index,
      'attr':attr,
      'weight':weight
    }

def ss_IW_gan_data_loader(data,tokenizer, max_len, batch_size):
  dataset = SS_IW_GAN_Data_loader(
    data,
    tokenizer,
    max_len
  )

  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
  ),dataset


  
from torch.utils.data import DataLoader, BatchSampler, RandomSampler,SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader

class SS_IW_Data_loader(Dataset):
  def __init__(self, data, tokenizer,max_len):

    self.Text = data['Text']
    self.Label = data['target']
    self.identity_gender_terms = data['gender']
    self.identity_race_terms = data['race']
    self.label_mask = data['label_mask']
    self.index = data['index']
    self.weights = data['weights']
    self.index = data['index']
    self.attr = data['attr']


    self.tokenizer = tokenizer
    self.max_len = max_len
    
  def __len__(self):
    return len(self.Text)

  def __getitem__(self, item):
    Text = str(self.Text[item])
    target = self.Label[item]
    identity_gender_term = self.identity_gender_terms[item]
    identity_race_term = self.identity_race_terms[item]
    label_mask = self.label_mask[item]
    index  = self.index[item]
    weights = self.weights[item]
    index  = self.index[item]
    attr  = self.attr[item]

    encoding = self.tokenizer.encode_plus(
      Text,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      truncation=True,
      return_tensors='pt',
    )
    return {
      'review_text': Text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long),
      'identity_gender_term': torch.tensor(identity_gender_term, dtype=torch.long),
      'identity_race_term':torch.tensor(identity_race_term, dtype=torch.long),
      'label_mask' : label_mask,
      'index':index,
      'weight':weights,
      'index':index,
      'attr':attr

    }

def create_IW_data_loader(data,tokenizer, max_len, batch_size):
  dataset = SS_IW_Data_loader(
    data,
    tokenizer,
    max_len
  )
  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
  ),dataset

def create_rand_batch_IW_data_loader(dataset, batch_size, num_batch =2):

  sampler = SubsetRandomSampler(list(range(num_batch*batch_size, (num_batch+1)*batch_size)))

  return DataLoader(dataset, sampler=sampler, shuffle=False, batch_size=batch_size, num_workers=1)



class SS_Fair_Data_loader(Dataset):
  def __init__(self, data, tokenizer,max_len):

    self.Text = data['Text']
    self.Label = data['target']
    self.identity_gender_terms = data['gender']
    self.identity_race_terms = data['race']
    self.label_mask = data['label_mask']
    self.index = data['index']
    self.attr = data['attr']

    self.tokenizer = tokenizer
    self.max_len = max_len
    
  def __len__(self):
    return len(self.Text)

  def __getitem__(self, item):
    Text = str(self.Text[item])
    target = self.Label[item]
    identity_gender_term = self.identity_gender_terms[item]
    identity_race_term = self.identity_race_terms[item]
    label_mask = self.label_mask[item]
    index  = self.index[item]
    attr  = self.attr[item]

    encoding = self.tokenizer.encode_plus(
      Text,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      truncation=True,
      return_tensors='pt',
    )
    return {
      'review_text': Text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long),
      'identity_gender_term': torch.tensor(identity_gender_term, dtype=torch.long),
      'identity_race_term':torch.tensor(identity_race_term, dtype=torch.long),
      'label_mask' : label_mask,
      'index':index,
      'attr':attr
    }

def create_Fair_data_loader(data,tokenizer, max_len, batch_size):
  dataset = SS_Fair_Data_loader(
    data,
    tokenizer,
    max_len
  )
  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
  ),dataset

def create_rand_batch_Fair_data_loader(dataset, batch_size, num_batch =2):

  sampler = RandomSampler(dataset,False,batch_size*num_batch )

  return DataLoader(dataset, sampler=sampler, shuffle=False, batch_size=batch_size, num_workers=1)








