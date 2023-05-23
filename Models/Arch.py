
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.ERROR)


import torch
import io
import random
import numpy as np
import time
import math
import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from collections import defaultdict



import torch.nn as nn
from transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix



# address
import sys
sys.path.append('/content/drive/MyDrive/Master_thesis/Metrics_Evaluations')
import metrics_eval

import sys
sys.path.append('/content/drive/MyDrive/Master_thesis_final/Dataloaders/HateXplain')
import Fair_GAN_BERT_data_loader

# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


#________________________________________________________________________________
epsilon = 1e-8
thres = 0.5
#________________________________________________________________________________

#------------------------------
#   The Generator as in
#   https://www.aclweb.org/anthology/2020.acl-main.191/
#   https://github.com/crux82/ganbert
#------------------------------

class BertClassifier(nn.Module):
  def __init__(self, n_classes):
    super(BertClassifier, self).__init__()
    self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    model_outputs = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask)
    hidden_states = model_outputs.last_hidden_state[:,0]

    output = self.drop(hidden_states)
    return hidden_states, self.out(output), F.softmax(self.out(output))



class Generator(nn.Module):
    def __init__(self, noise_size=100, output_size=512, hidden_sizes=[512], dropout_rate=0.1):
        super(Generator, self).__init__()
        layers = []
        hidden_sizes = [noise_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])
        #hidden_sizes[-1] = 512
        layers.append(nn.Linear(hidden_sizes[-1],output_size))
        self.layers = nn.Sequential(*layers)
        print("G:layers",layers)
        print("G:self.layers", self.layers)

    def forward(self, noise):
        output_rep = self.layers(noise)
        return output_rep


#------------------------------
#   The Discriminator
#   https://www.aclweb.org/anthology/2020.acl-main.191/
#   https://github.com/crux82/ganbert
#------------------------------
class Discriminator(nn.Module):
    def __init__(self, input_size=512, hidden_sizes=[512], num_labels=2, dropout_rate=0.1):
        super(Discriminator, self).__init__()

        self.input_dropout = nn.Dropout(p=dropout_rate)
        layers = []
        hidden_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

        self.layers = nn.Sequential(*layers) #per il flatten
        self.logit = nn.Linear(hidden_sizes[-1],num_labels+1) # +1 for the probability of this sample being fake/real.
        self.softmax = nn.Softmax(dim=-1)

        self.relu= nn.LeakyReLU(0.2, inplace=True)



    def forward(self, input_rep, label = None,label_embed = None):


        input_rep = self.input_dropout(input_rep)
        last_rep = self.layers(input_rep)
        logits = self.logit(last_rep)
        probs = self.softmax(logits)


        return last_rep, logits, probs



class Adversary(nn.Module):
    def __init__(self,seed_val, identity_labels = 2):
        super(Adversary, self).__init__()

        self.a1 = nn.Linear(768,240)
        self.a2 = nn.Linear(240, identity_labels)

        self.a1_r = nn.Linear(768,240)
        self.a2_r = nn.Linear(240, identity_labels)


        torch.manual_seed(seed_val)
        nn.init.xavier_normal_(self.a1.weight)
        nn.init.xavier_normal_(self.a1_r.weight)


    def forward(self, input_ids):

        adversary = F.relu(self.a1(input_ids))
        adversary_output_gender = self.a2(adversary)

        adversary_r = F.relu(self.a1_r(input_ids))
        adversary_output_race = self.a2_r(adversary_r)


        return adversary_output_gender,adversary_output_race



class Adversary1(nn.Module):
    def __init__(self,seed_val, identity_labels = 2):
        super(Adversary1, self).__init__()

        self.a1 = nn.Linear(768,240)
        self.a2 = nn.Linear(240, identity_labels)
        self.a3 = nn.Linear(240, identity_labels)


        torch.manual_seed(seed_val)
        nn.init.xavier_normal_(self.a1.weight)


    def forward(self, input_ids):

        adversary = F.relu(self.a1(input_ids))
        adversary_output_gender = self.a2(adversary)
        adversary_output_race = self.a3(adversary)


        return adversary_output_gender,adversary_output_race



def noise_gen(input_shape, noise_size, device):

  return torch.zeros(input_shape,noise_size, device=device).uniform_(0, 1)



class Adversary_shared(nn.Module):
    def __init__(self,seed_val, identity_labels = 2):
        super(Adversary_shared, self).__init__()

        self.a1 = nn.Linear(768,240)
        self.a2 = nn.Linear(240, identity_labels)
        self.a3 = nn.Linear(240, identity_labels)




        torch.manual_seed(seed_val)
        nn.init.xavier_normal_(self.a1.weight)


    def forward(self, input_ids):

        adversary = F.relu(self.a1(input_ids))
        adversary_output_gender = self.a2(adversary)
        adversary_output_race = self.a3(adversary)


        return adversary_output_gender,adversary_output_race



def supervised_loss_IW(D_real_logits,b_labels,b_label_mask,weight):

  # Disciminator's LOSS estimation
  logits = D_real_logits[:,0:-1]
  log_probs = F.log_softmax(logits, dim=-1)
  label2one_hot = torch.nn.functional.one_hot(b_labels, 3)
  per_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)
  per_example_loss = weight*per_example_loss
  per_example_loss = torch.masked_select(per_example_loss, b_label_mask.to(device))
  per_example_w = torch.masked_select(weight, b_label_mask.to(device))
  labeled_example_count = per_example_loss.type(torch.float32).numel()
  labeled_example_count = labeled_example_count*torch.sum(per_example_w)

  if labeled_example_count == 0:
    D_L_Supervised = 0
  else:
    D_L_Supervised = torch.div(torch.sum(per_example_loss.to(device)), labeled_example_count)

  return D_L_Supervised



def noise_gen(input_shape, noise_size, device):

  return torch.zeros(input_shape,noise_size, device=device).uniform_(0, 1)

def supervised_loss(D_real_logits,b_labels,b_label_mask):

  # Disciminator's LOSS estimation
  logits = D_real_logits[:,0:-1]
  # print("logits",logits.shape)
  log_probs = F.log_softmax(logits, dim=-1)
  # print("D_real_logits",D_real_logits.shape)
  # print("logits",logits.shape)

  label2one_hot = torch.nn.functional.one_hot(b_labels, 3)
  
  per_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)

  per_example_loss = torch.masked_select(per_example_loss, b_label_mask.to(device))
  
  labeled_example_count = per_example_loss.type(torch.float32).numel()
  
  if labeled_example_count == 0:
    D_L_Supervised = 0
  else:
    D_L_Supervised = torch.div(torch.sum(per_example_loss.to(device)), labeled_example_count)

  return D_L_Supervised



loss_criterion =  torch.nn.CrossEntropyLoss()
loss_fn =  torch.nn.CrossEntropyLoss()
