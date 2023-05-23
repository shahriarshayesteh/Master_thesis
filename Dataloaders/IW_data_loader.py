'''
The most important reason that we do not go with fine terms for IW 
is that in general we do not have any idea of about all the words 
that represents genders or different races in general in the sentences.
Also, we do not know how these words are distributed. For example, 
a sentence like minority from group A are discriminated by groupB. 
In general we know that sentence is about race either dataset have a target group or we mine it 
but in two case we miss terms that are not representitive 
words about race like white,black, arab, etc and this cause that the featuer matrix Z does not have accurate featuers.
Also, knowing about the terms related to race or gender mentioned in the big datasets for  hate detection task for example is not easy to guess 
and it's changing over time. Therefore, we thought if we go by coarse terms in general in oppse to fin terms, the result of IW is more reliable if it's to be used in future in any system.
'''

import torch
import io
import torch.nn.functional as F
import random
import numpy as np
import time
import math
import datetime
import torch.nn as nn
from transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,precision_score,recall_score,f1_score
from collections import defaultdict


#!pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
#!pip install sentencepiece
import pandas as pd
##Set random values
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(seed_val)
  
  
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
    
#address
import sys
sys.path.append('/content/drive/MyDrive/Master_thesis/Metrics_Evaluations')
import metrics_eval
    
from numpy.core.getlimits import inf
import re
import os
import numpy as np
import pandas as pd
# from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_predict





def sentence_len(text):
    text = [i.lower() for i in text]
    filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n'
    maketrans = str.maketrans
    translate_map = maketrans(filters, " " * len(filters))
    text = [i.translate(translate_map) for i in text]
    text = np.array([(" " + " ".join(i.strip().split()) + " ") for i in text])
    sen_length = np.array([len(i.split()) for i in text])
    return sen_length


def weight_output(data, normalized = False):

  index_list = list(data.index)
  trr= data.iloc[index_list][['Text']].copy()
  print(trr.Text)

  sen_len  = sentence_len(trr.Text)
  
  if normalized:
    data['sen_len'] = sen_len/sum(sen_len)
  else:
    data['sen_len'] = sen_len



  return np.array(data[['gender','race','sen_len']])




def instance_weighting(Z,target,dir_processed = '/content/drive/MyDrive/Master_thesis/Debiasing_Methods/Instance_weghting_files/',use_loaded = False):


      clf = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=233, n_jobs=3, criterion='entropy')

      y_pred = cross_val_predict(clf, Z, target, cv=200, n_jobs=1, method='predict_proba')

      print('Refit log loss: %.5f' % (log_loss(target, y_pred[:, 1])))


      acc_maj = max(1 - sum(target) / len(target), sum(target) / len(target))
      # print(roc_auc_score(to_categorical(y), y_pred))
      print(accuracy_score(target, np.argmax(y_pred, 1)), acc_maj)

      # take the predictive probability that is correspond to the true label
      # for i in range(len(target)):
      #   print(y_pred)
      #   print(int(target[i]))
      #   print(y_pred[i, target[i]])

      propensity = np.array([y_pred[i, int(target[i])] for i in range(len(target))])
      print("propensity",propensity)
      np.save(dir_processed + "propensity.npy", propensity)

      weights = 1 / propensity
      print("weights",weights)

      # mean of weights of samples that has label as non-toxic
      sd = np.array([weights[i] for i in range(len(weights)) if target[i] == 0])
      a = np.mean(np.array([weights[i] for i in range(len(weights)) if target[i] == 0 if weights[i] != np.inf ]))
      print("a",a)
      # mean of weights of samples that has label as toxic
      b = np.mean(np.array([weights[i] for i in range(len(weights)) if target[i] == 1 if weights[i] != np.inf ]))
      # print((1 / a) / (1 / a + 1 / b), (1 / b) / (1 / a + 1 / b))
      print("b",b)




      weights = np.array([(weights[i] / a if target[i] == 0 else weights[i] / b) for i in range(len(weights))])
      print("weights",weights)

      weights /= weights.mean()
      print("weights",weights)


      np.save(dir_processed + "weights.npy", weights)
      # np.save(dir_processed + "weights_train.npy", weights[:len(train_data)])
      # np.save(dir_processed + "weights_dev.npy", weights[len(train_data):len(train_data) + len(valid_data)])
      return weights,sd,y_pred