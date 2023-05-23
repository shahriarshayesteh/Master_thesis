import random
import pandas as pd
import numpy as np


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from torch.utils.data import DataLoader, BatchSampler, RandomSampler,SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import torch

from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.ERROR)


import sys
sys.path.append('/content/drive/MyDrive/SS_Fair/Dataloaders')
import IW_data_loader

import sys
sys.path.append('/content/drive/MyDrive/Master_thesis/Metrics_Evaluations')
import metrics_eval

# Data Samplers

class SS_Data_Sampler:
  '''
  This class generate a subsample of labeled and unlabeled data distributed based on class label and protected label {race , gender} (non-overlapping samples).

  The sample distribution is given as two class attributes to each SS_Data_Sampler object. It is important to note,
  each object generated with a fixed random-seed value and it can beused to generate propoer training data for all the models.

  '''

  def __init__(self,data,non_toxic_proportion_list,toxic_proportion_list,seed_value):

    '''
    data = training data

    non_toxic_proportion_list = [non_toxic_non_protected_label, non_toxic_non_protected_unlabel,non_toxic_protected_race_label,
                              non_toxic_protected_race_unlabel,non_toxic_protected_gender_label,non_toxic_protected_gender_unlabel]

    toxic_proportion_list = [toxic_non_protected_label, toxic_non_protected_unlabel,toxic_protected_race_label,
                           toxic_protected_race_unlabel,toxic_protected_gender_label,toxic_protected_gender_unlabel]
    seed_value = to set a random seed

    '''

    #class attributes
    self.data =data
    self.non_toxic_proportion_list = non_toxic_proportion_list
    self.toxic_proportion_list = toxic_proportion_list
    self.seed_value = seed_value

    # generate a fix set of index to sample data from training
    # and assing them to class attributes to make generate the same subsample from an class object over and over
    self.index_sampler()


  def index_sampler(self):
    '''
    generate a fix set of index to sample data from training
    and assing them to class attributes to make generate the same subsample from an class object over and over
    '''
    # get the index of subsamples
    index_non_toxic_non_protected_0 = list(self.data.query('Label == 0 and `gender` == 0 and `race` == 0').index)
    index_non_toxic_protected_race = list(self.data.query('Label == 0 and `gender` == 0 and `race` == 1').index)
    index_non_toxic_protected_gender= list(self.data.query('Label == 0 and `gender` == 1 and `race` == 0').index)
    #index_non_toxic_protected_gender_race= list(self.data.query('Label == 0 and `gender` == 1 and `race` == 1').index)

    index_toxic_non_protected_0 = list(self.data.query('Label == 1 and `gender` == 0 and `race` == 0').index)
    index_toxic_non_protected_race = list(self.data.query('Label == 1 and `gender` == 0 and `race` == 1').index)
    index_toxic_non_protected_gender = list(self.data.query('Label == 1 and `gender` == 1 and `race` == 0').index)
    #index_toxic_protected_gender_race= list(self.data.query('Label == 1 and `gender` == 1 and `race` == 1').index)


    #index of the sampled data with a fixed random seed
    random.seed(self.seed_value)
    # self.index_non_toxic_non_protected_0_label = random.sample(index_non_toxic_non_protected_0, self.non_toxic_proportion_list[0])
    # index_non_toxic_non_protected_0= SS_Data_Sampler.remove_intersection(index_non_toxic_non_protected_0,self.index_non_toxic_non_protected_0_label)
    # self.index_non_toxic_non_protected_0_unlabel =  random.sample(index_non_toxic_non_protected_0, self.non_toxic_proportion_list[1])

    # self.index_non_toxic_protected_race_label =  random.sample(index_non_toxic_protected_race, self.non_toxic_proportion_list[2])
    # index_non_toxic_protected_race= SS_Data_Sampler.remove_intersection(index_non_toxic_protected_race,self.index_non_toxic_protected_race_label)
    # self.index_non_toxic_protected_race_unlabel =  random.sample(index_non_toxic_protected_race, self.non_toxic_proportion_list[3])

    # self.index_non_toxic_protected_gender_label =  random.sample(index_non_toxic_protected_gender, self.non_toxic_proportion_list[4])
    # index_non_toxic_protected_gender= SS_Data_Sampler.remove_intersection(index_non_toxic_protected_gender,self.index_non_toxic_protected_gender_label)
    # self.index_non_toxic_protected_gender_unlabel=  random.sample(index_non_toxic_protected_gender, self.non_toxic_proportion_list[5])

    # self.index_toxic_non_protected_0_label =  random.sample(index_toxic_non_protected_0, self.toxic_proportion_list[0])
    # index_toxic_non_protected_0= SS_Data_Sampler.remove_intersection(index_toxic_non_protected_0,self.index_toxic_non_protected_0_label)
    # self.index_toxic_non_protected_0_unlabel =  random.sample(index_toxic_non_protected_0, self.toxic_proportion_list[1])

    # self.index_toxic_non_protected_race_label =  random.sample(index_toxic_non_protected_race, self.toxic_proportion_list[2])
    # index_toxic_non_protected_race= SS_Data_Sampler.remove_intersection(index_toxic_non_protected_race,self.index_toxic_non_protected_race_label)
    # self.index_toxic_non_protected_race_unlabel =  random.sample(index_toxic_non_protected_race, self.toxic_proportion_list[3])

    # self.index_toxic_non_protected_gender_label =  random.sample(index_toxic_non_protected_gender, self.toxic_proportion_list[4])
    # index_toxic_non_protected_gender= SS_Data_Sampler.remove_intersection(index_toxic_non_protected_gender,self.index_toxic_non_protected_gender_label)
    # self.index_toxic_non_protected_gender_unlabel =  random.sample(index_toxic_non_protected_gender, self.toxic_proportion_list[5])

    # add for overlapping scenario, add both group overlapping samples as the last two in the input list

    # self.index_non_toxic_protected_gender_race_label =  random.sample(index_non_toxic_protected_gender_race, self.non_toxic_proportion_list[6])
    # index_non_toxic_protected_gender_race= SS_Data_Sampler.remove_intersection(index_non_toxic_protected_gender_race,self.index_non_toxic_protected_gender_race_label)
    # self.index_non_toxic_protected_gender_race_unlabel=  random.sample(index_non_toxic_protected_gender_race, self.non_toxic_proportion_list[7])


    # self.index_toxic_protected_gender_race_label =  random.sample(index_toxic_protected_gender_race, self.toxic_proportion_list[6])
    # index_toxic_protected_gender_race= SS_Data_Sampler.remove_intersection(index_toxic_protected_gender_race,self.index_toxic_protected_gender_race_label)
    # self.index_toxic_protected_gender_race_unlabel =  random.sample(index_toxic_protected_gender_race, self.toxic_proportion_list[7])


    random.seed(self.seed_value)
    print("index_non_toxic_non_protected_0",len(index_non_toxic_non_protected_0))
    print("self.non_toxic_proportion_list[0]",self.non_toxic_proportion_list[0])
    self.index_non_toxic_non_protected_0_label = SS_Data_Sampler.custom_sample(index_non_toxic_non_protected_0, self.non_toxic_proportion_list[0], self.seed_value)
    index_non_toxic_non_protected_0 = SS_Data_Sampler.remove_intersection(index_non_toxic_non_protected_0, self.index_non_toxic_non_protected_0_label)
    print("index_non_toxic_non_protected_0",len(index_non_toxic_non_protected_0),index_non_toxic_non_protected_0)
    print("self.non_toxic_proportion_list[1]",self.non_toxic_proportion_list[1])
    self.index_non_toxic_non_protected_0_unlabel = SS_Data_Sampler.custom_sample(index_non_toxic_non_protected_0, self.non_toxic_proportion_list[1], self.seed_value)

    self.index_non_toxic_protected_race_label = SS_Data_Sampler.custom_sample(index_non_toxic_protected_race, self.non_toxic_proportion_list[2], self.seed_value)
    index_non_toxic_protected_race = SS_Data_Sampler.remove_intersection(index_non_toxic_protected_race, self.index_non_toxic_protected_race_label)
    self.index_non_toxic_protected_race_unlabel = SS_Data_Sampler.custom_sample(index_non_toxic_protected_race, self.non_toxic_proportion_list[3], self.seed_value)

    self.index_non_toxic_protected_gender_label = SS_Data_Sampler.custom_sample(index_non_toxic_protected_gender, self.non_toxic_proportion_list[4], self.seed_value)
    index_non_toxic_protected_gender = SS_Data_Sampler.remove_intersection(index_non_toxic_protected_gender, self.index_non_toxic_protected_gender_label)
    self.index_non_toxic_protected_gender_unlabel = SS_Data_Sampler.custom_sample(index_non_toxic_protected_gender, self.non_toxic_proportion_list[5], self.seed_value)

    self.index_toxic_non_protected_0_label = SS_Data_Sampler.custom_sample(index_toxic_non_protected_0, self.toxic_proportion_list[0], self.seed_value)
    index_toxic_non_protected_0 = SS_Data_Sampler.remove_intersection(index_toxic_non_protected_0, self.index_toxic_non_protected_0_label)
    self.index_toxic_non_protected_0_unlabel = SS_Data_Sampler.custom_sample(index_toxic_non_protected_0, self.toxic_proportion_list[1], self.seed_value)    

    self.index_toxic_non_protected_race_label = SS_Data_Sampler.custom_sample(index_toxic_non_protected_race, self.toxic_proportion_list[2], self.seed_value)
    print("self.toxic_proportion_list[2]",self.toxic_proportion_list[2])
    print("index_toxic_non_protected_race",len(index_toxic_non_protected_race))
    print("self.index_toxic_non_protected_race_label",len(self.index_toxic_non_protected_race_label))

    index_toxic_non_protected_race = SS_Data_Sampler.remove_intersection(index_toxic_non_protected_race, self.index_toxic_non_protected_race_label)
    print("index_toxic_non_protected_race",len(index_toxic_non_protected_race))
    self.index_toxic_non_protected_race_unlabel = SS_Data_Sampler.custom_sample(index_toxic_non_protected_race, self.toxic_proportion_list[3], self.seed_value)

    self.index_toxic_non_protected_gender_label = SS_Data_Sampler.custom_sample(index_toxic_non_protected_gender, self.toxic_proportion_list[4], self.seed_value)
    index_toxic_non_protected_gender = SS_Data_Sampler.remove_intersection(index_toxic_non_protected_gender, self.index_toxic_non_protected_gender_label)
    self.index_toxic_non_protected_gender_unlabel = SS_Data_Sampler.custom_sample(index_toxic_non_protected_gender, self.toxic_proportion_list[5], self.seed_value)




  def custom_sample(population, subsample_size, random_state):
    random.seed(random_state)

    print("custom",  subsample_size,len(population))

    if subsample_size <= len(population):
        return random.sample(population, subsample_size)
    else:

        return [random.choice(population) for _ in range(subsample_size)]

  def data_sampler(self, GANBERT = False, IW = False):

    '''
    it samples data based on index of samples randomly generated by def index_sampler for a subset of data.
    '''

    if IW:
      # if we use instance weighting preprocessing method, compute and add the weights as a columns to the training data
      self.instance_weight()


    # sampled data from the dataset with the index found before
    if GANBERT:

        non_toxic_non_protected_0_label = self.data.iloc[self.index_non_toxic_non_protected_0_label]
        non_toxic_non_protected_0_unlabel = self.data.iloc[self.index_non_toxic_non_protected_0_unlabel]

        non_toxic_protected_race_label = self.data.iloc[self.index_non_toxic_protected_race_label]
        non_toxic_protected_race_unlabel= self.data.iloc[self.index_non_toxic_protected_race_unlabel]

        non_toxic_protected_gender_label = self.data.iloc[self.index_non_toxic_protected_gender_label]
        non_toxic_protected_gender_unlabel = self.data.iloc[self.index_non_toxic_protected_gender_unlabel]

        non_toxic_non_protected_0_label['target'] = 1
        non_toxic_protected_race_label['target']=  1
        non_toxic_protected_gender_label['target']= 1

        non_toxic_non_protected_0_label['label_mask'] = True
        non_toxic_protected_race_label['label_mask']=  True
        non_toxic_protected_gender_label['label_mask']= True

        non_toxic_non_protected_0_unlabel['target'] = 0
        non_toxic_protected_race_unlabel['target']=  0
        non_toxic_protected_gender_unlabel['target']= 0

        non_toxic_non_protected_0_unlabel['label_mask'] = False
        non_toxic_protected_race_unlabel['label_mask']=  False
        non_toxic_protected_gender_unlabel['label_mask']= False

        toxic_non_protected_0_label  = self.data.iloc[self.index_toxic_non_protected_0_label]
        toxic_non_protected_0_unlabel = self.data.iloc[self.index_toxic_non_protected_0_unlabel]

        toxic_protected_race_label = self.data.iloc[self.index_toxic_non_protected_race_label]
        toxic_protected_race_unlabel = self.data.iloc[self.index_toxic_non_protected_race_unlabel]

        toxic_protected_gender_label = self.data.iloc[self.index_toxic_non_protected_gender_label]
        toxic_protected_gender_unlabel = self.data.iloc[self.index_toxic_non_protected_gender_unlabel]

        toxic_non_protected_0_label['target']= 2
        toxic_protected_race_label['target']= 2
        toxic_protected_gender_label['target']=2

        toxic_non_protected_0_label['label_mask']= True
        toxic_protected_race_label['label_mask']= True
        toxic_protected_gender_label['label_mask']=True

        toxic_non_protected_0_unlabel['target'] = 0
        toxic_protected_race_unlabel['target']=  0
        toxic_protected_gender_unlabel['target']= 0


        toxic_non_protected_0_unlabel['label_mask'] = False
        toxic_protected_race_unlabel['label_mask']=  False
        toxic_protected_gender_unlabel['label_mask']= False

        fair_train_label = pd.concat([non_toxic_non_protected_0_label,non_toxic_protected_race_label,non_toxic_protected_gender_label,
                                  toxic_non_protected_0_label,toxic_protected_race_label,toxic_protected_gender_label]).reset_index(drop=True)

        fair_train_unlabel = pd.concat([non_toxic_non_protected_0_unlabel,non_toxic_protected_race_unlabel,non_toxic_protected_gender_unlabel,
                                  toxic_non_protected_0_unlabel,toxic_protected_race_unlabel,toxic_protected_gender_unlabel]).reset_index(drop=True)

        fair_train =  pd.concat([fair_train_label,fair_train_unlabel])
        fair_train = fair_train.reset_index(drop=True)
        return fair_train

    else:


        non_toxic_non_protected_0_label = self.data.iloc[self.index_non_toxic_non_protected_0_label]
        non_toxic_non_protected_0_unlabel = self.data.iloc[self.index_non_toxic_non_protected_0_unlabel]

        non_toxic_protected_race_label = self.data.iloc[self.index_non_toxic_protected_race_label]
        non_toxic_protected_race_unlabel= self.data.iloc[self.index_non_toxic_protected_race_unlabel]

        non_toxic_protected_gender_label = self.data.iloc[self.index_non_toxic_protected_gender_label]
        non_toxic_protected_gender_unlabel = self.data.iloc[self.index_non_toxic_protected_gender_unlabel]

        non_toxic_non_protected_0_label['target'] = 0
        non_toxic_protected_race_label['target']=  0
        non_toxic_protected_gender_label['target']= 0

        non_toxic_non_protected_0_label['label_mask'] = True
        non_toxic_protected_race_label['label_mask']=  True
        non_toxic_protected_gender_label['label_mask']= True

        non_toxic_non_protected_0_unlabel['target'] = -1
        non_toxic_protected_race_unlabel['target']=  -1
        non_toxic_protected_gender_unlabel['target']= -1

        non_toxic_non_protected_0_unlabel['label_mask'] = False
        non_toxic_protected_race_unlabel['label_mask']=  False
        non_toxic_protected_gender_unlabel['label_mask']= False

        toxic_non_protected_0_label  = self.data.iloc[self.index_toxic_non_protected_0_label]
        toxic_non_protected_0_unlabel = self.data.iloc[self.index_toxic_non_protected_0_unlabel]

        toxic_protected_race_label = self.data.iloc[self.index_toxic_non_protected_race_label]
        toxic_protected_race_unlabel = self.data.iloc[self.index_toxic_non_protected_race_unlabel]

        toxic_protected_gender_label = self.data.iloc[self.index_toxic_non_protected_gender_label]
        toxic_protected_gender_unlabel = self.data.iloc[self.index_toxic_non_protected_gender_unlabel]

        toxic_non_protected_0_label['target']= 1
        toxic_protected_race_label['target']= 1
        toxic_protected_gender_label['target']=1

        toxic_non_protected_0_label['label_mask']= True
        toxic_protected_race_label['label_mask']= True
        toxic_protected_gender_label['label_mask']=True

        toxic_non_protected_0_unlabel['target'] = -1
        toxic_protected_race_unlabel['target']=  -1
        toxic_protected_gender_unlabel['target']= -1


        toxic_non_protected_0_unlabel['label_mask'] = False
        toxic_protected_race_unlabel['label_mask']=  False
        toxic_protected_gender_unlabel['label_mask']= False

        fair_train_label = pd.concat([non_toxic_non_protected_0_label,non_toxic_protected_race_label,non_toxic_protected_gender_label,
                                  toxic_non_protected_0_label,toxic_protected_race_label,toxic_protected_gender_label]).reset_index(drop=True)

        fair_train_unlabel = pd.concat([non_toxic_non_protected_0_unlabel,non_toxic_protected_race_unlabel,non_toxic_protected_gender_unlabel,
                                  toxic_non_protected_0_unlabel,toxic_protected_race_unlabel,toxic_protected_gender_unlabel]).reset_index(drop=True)


        # return fair_train_label,fair_train_unlabel
        return fair_train_label


  def instance_weight(self):
    "Change the name of the class"
    #address
    dd = IW_data_loader.weight_output(self.data)
    weight_tr,sd,y_pred = IW_data_loader.instance_weighting(dd,list(self.data.Label))
    self.data['weights'] = weight_tr

  #instance_weight1 is static method eq to instance_weight and it is defined
  #for use for validation\test data which is not object of this class
  def instance_weight1(data):
    "Change the name of the class"
    dd = IW_data_loader.weight_output(data)
    weight_tr,sd,y_pred = IW_data_loader.instance_weighting(dd,list(data.Label))
    data['weights'] = weight_tr
    return data

  #remove the samples index from the subsample list that are already taken from the dataset.
  def remove_intersection(l1, l2):
    arr1 = np.array(l1)
    arr2 = np.array(l2)
    mask = np.in1d(arr1, arr2, invert=True)
    return arr1[mask].tolist()

  #add the proper column to validation set

  def validator_target(validation, GANBERT = False,IW = False):
    if IW:
     validation = SS_Data_Sampler.instance_weight1(validation)

    if GANBERT:

      validation =validation.reset_index(drop=True)
      validation['target'] = validation['Label'] +1
      validation['label_mask'] = True
      return validation



    else:
      validation =validation.reset_index(drop=True)
      validation['target'] = validation['Label']
      validation['label_mask'] = True
      return validation


# def FairModelDataLoader:
#   '''
#   This is supposed to be a supper class for all the models data_loader
#   '''
