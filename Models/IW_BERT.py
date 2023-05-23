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
from sklearn.metrics import matthews_corrcoef

from numpy.core.getlimits import inf
import re
import os
import numpy as np
import pandas as pd
# from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_predict
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.ERROR)

import sys
sys.path.append('/content/drive/MyDrive/SS_Fair/Models/')
import Fair_NDABERT


from transformers import DistilBertTokenizer, DistilBertModel

#!pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
#!pip install sentencepiece
import pandas as pd
##Set random values
# seed_val = 42
# random.seed(seed_val)
# np.random.seed(seed_val)
# torch.manual_seed(seed_val)
# if torch.cuda.is_available():
#   torch.cuda.manual_seed_all(seed_val)



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


import sys
sys.path.append('/content/drive/MyDrive/Master_thesis_final/Dataloaders/HateXplain')
import adversarial_self_training_bert_loader

import sys
sys.path.append('/content/drive/MyDrive/Master_thesis/Metrics_Evaluations')
import metrics_eval


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
    return self.out(output), F.softmax(self.out(output))








thres = 0.5
import sys
sys.path.append('/content/drive/MyDrive/Master_thesis/Metrics_Evaluations')
import metrics_eval

class Bert_trainer(Fair_NDABERT.Pre_training_stage):

  def __init__(self,model,train_loader, validation_loader, loss,optimizer_lr= 1e-5, epoch= 10,path ="", model_name = "",transformer_name = ""):

    self.model = model
    self.train_loader = train_loader
    self.validation_loader = validation_loader
    self.loss = loss
    self.optimizer = AdamW(self.model.parameters(), lr=optimizer_lr, correct_bias=False)
    self.epoch = epoch
    self.path = path
    self.model_name = model_name
    self.transformer_name = transformer_name

  def train(self):

   losses = []
   for data in self.train_loader:

      input_id = data["input_ids"].to(device)
      attention_mask = data["attention_mask"].to(device)
      target = data["targets"].to(device)
      weights = data["weight"].to(device)


      logit, prob = self.model(input_id,attention_mask)

      loss = self.loss(logit,target)
      # print("loss", loss)
      loss = loss*weights
      loss = torch.mean(loss)
      losses.append(loss.item())

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

   print("loss for epoch",sum(losses)/len(losses))

  def conduct_validation(self,data_loader):

    eval_loss, eval_balanced_accuracy, eval_accuracy, eval_precision, eval_recall, eval_f1 = 0,0, 0, 0, 0, 0
    nb_eval_steps = 0

    predictions_net = np.empty((0,))
    truths = np.empty((0,))
    identities_gender = np.empty((0,))
    identities_race = np.empty((0,))

    correct_net = 0
    total = 0

    self.model.eval()
    with torch.no_grad(): # IMPORTANT: we don't want to do back prop during validation/testing!
      for idx, data in enumerate(data_loader):

        # get the inputs and labels
        text,input_ids, attention_mask, targets,identity_term_gender,identity_term_race,label_mask,weight,index,attr = data.values()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)
        identity_term_gender = identity_term_gender.to(device)
        identity_term_race = identity_term_race.to(device)
        weight = weight.to(device)


        logit, prob = self.model(input_ids,attention_mask)
        confidence, predicted_label = torch.max(prob, 1)


        batch_size = targets.size(0)
        total += batch_size
        correct_net_batch = (predicted_label == targets).sum().item()
        correct_net += correct_net_batch


        predictions_net = np.concatenate((predictions_net, predicted_label.cpu().numpy()))
        truths = np.concatenate((truths, targets.cpu().numpy()))
        identities_gender = np.concatenate((identities_gender, identity_term_gender.cpu().numpy()))
        identities_race = np.concatenate((identities_race, identity_term_race.cpu().numpy()))


        pred = predicted_label.detach().cpu().numpy()
        label_ids = targets.to('cpu').numpy()

        tmp_eval_balanced_accuracy,tmp_eval_accuracy, tmp_eval_precision, temp_eval_recall, tmp_eval_f1 = metrics_eval.Performance_Metrics.get_metrics(label_ids, pred)

        eval_accuracy += tmp_eval_accuracy
        eval_precision += tmp_eval_precision
        eval_recall += temp_eval_recall
        eval_f1 += tmp_eval_f1
        nb_eval_steps += 1
        eval_balanced_accuracy += tmp_eval_balanced_accuracy

    f1_score = eval_f1/nb_eval_steps
    prec_score = eval_precision/nb_eval_steps
    recall_score = eval_recall/nb_eval_steps
    acc_score = eval_accuracy/nb_eval_steps
    balanced_acc_score = eval_balanced_accuracy/nb_eval_steps
    accuracy_metrics_dic = {
        "f1": f1_score,
        "Precision": prec_score,
        "Recall": recall_score,
        "acc": acc_score,
      "balance_acc": balanced_acc_score
    }

    print("F1 Score: ", f1_score)
    print("Precision Score: ", prec_score)
    print("Recall Score: ", recall_score)
    print("Acc Score: ", acc_score, "\n\n")

    return (predictions_net, truths, identities_gender,identities_race, accuracy_metrics_dic)

  def conduct_validation_ippts(self,data_loader):

    eval_loss, eval_balanced_accuracy, eval_accuracy, eval_precision, eval_recall, eval_f1 = 0,0, 0, 0, 0, 0
    nb_eval_steps = 0

    predictions_net = np.empty((0,))
    truths = np.empty((0,))
    identities_gender = np.empty((0,))
    identities_race = np.empty((0,))
    fine_terms = []


    correct_net = 0
    total = 0
    self.model.eval()

    with torch.no_grad(): # IMPORTANT: we don't want to do back prop during validation/testing!

      for batch in data_loader:

        # Unpack this training batch from our dataloader.
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)

        identity_term_gender = batch['identity_gender_term'].to(device)
        identity_term_race = batch['identity_race_term'].to(device)
        fine_term = batch['attr']


        logit, prob = self.model(b_input_ids,b_input_mask)
        confidence, predicted_label = torch.max(prob, 1)


        batch_size = targets.size(0)
        total += batch_size
        correct_net_batch = (predicted_label == targets).sum().item()
        correct_net += correct_net_batch


        predictions_net = np.concatenate((predictions_net, predicted_label.cpu().numpy()))
        truths = np.concatenate((truths, targets.cpu().numpy()))
        identities_gender = np.concatenate((identities_gender, identity_term_gender.cpu().numpy()))
        identities_race = np.concatenate((identities_race, identity_term_race.cpu().numpy()))
        fine_terms += fine_term



        pred = predicted_label.detach().cpu().numpy()
        label_ids = targets.to('cpu').numpy()

        label_ids1 = label_ids -1
        pred1 = pred

        tmp_eval_balanced_accuracy,tmp_eval_accuracy, tmp_eval_precision, temp_eval_recall, tmp_eval_f1 = metrics_eval.Performance_Metrics.get_metrics(label_ids1, pred1)

        eval_accuracy += tmp_eval_accuracy
        eval_precision += tmp_eval_precision
        eval_recall += temp_eval_recall
        eval_f1 += tmp_eval_f1
        nb_eval_steps += 1
        eval_balanced_accuracy += tmp_eval_balanced_accuracy


    f1_score = eval_f1/nb_eval_steps
    prec_score = eval_precision/nb_eval_steps
    recall_score = eval_recall/nb_eval_steps
    acc_score = eval_accuracy/nb_eval_steps
    balanced_acc_score = eval_balanced_accuracy/nb_eval_steps


    accuracy_metrics_dic = {
        "f1": f1_score,
        "Precision": prec_score,
        "Recall": recall_score,
        "acc": acc_score,
      "balance_acc": balanced_acc_score
    }

    print("F1 Score: ", f1_score)
    print("Precision Score: ", prec_score)
    print("Recall Score: ", recall_score)
    print("Acc Score: ", acc_score, "\n\n")

    # self.classifier.train()

    return (predictions_net, truths, identities_gender,identities_race, fine_terms,accuracy_metrics_dic)


  def conduct_validation_fine_term(self,data_loader):

    eval_loss, eval_balanced_accuracy, eval_accuracy, eval_precision, eval_recall, eval_f1 = 0,0, 0, 0, 0, 0
    nb_eval_steps = 0

    predictions_net = np.empty((0,))
    truths = np.empty((0,))
    identities_gender = np.empty((0,))
    identities_race = np.empty((0,))
    fine_terms = []


    correct_net = 0
    total = 0
    self.model.eval()

    with torch.no_grad(): # IMPORTANT: we don't want to do back prop during validation/testing!

      for batch in data_loader:

        # Unpack this training batch from our dataloader.
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)

        identity_term_gender = batch['identity_gender_term'].to(device)
        identity_term_race = batch['identity_race_term'].to(device)
        fine_term = batch['attr']


        logit, prob = self.model(b_input_ids,b_input_mask)
        confidence, predicted_label = torch.max(prob, 1)


        batch_size = targets.size(0)
        total += batch_size
        correct_net_batch = (predicted_label == targets).sum().item()
        correct_net += correct_net_batch


        predictions_net = np.concatenate((predictions_net, predicted_label.cpu().numpy()))
        truths = np.concatenate((truths, targets.cpu().numpy()))
        identities_gender = np.concatenate((identities_gender, identity_term_gender.cpu().numpy()))
        identities_race = np.concatenate((identities_race, identity_term_race.cpu().numpy()))
        fine_terms.append(fine_term)



        pred = predicted_label.detach().cpu().numpy()
        label_ids = targets.to('cpu').numpy()

        label_ids1 = label_ids -1
        pred1 = pred

        tmp_eval_balanced_accuracy,tmp_eval_accuracy, tmp_eval_precision, temp_eval_recall, tmp_eval_f1 = metrics_eval.Performance_Metrics.get_metrics(label_ids1, pred1)

        eval_accuracy += tmp_eval_accuracy
        eval_precision += tmp_eval_precision
        eval_recall += temp_eval_recall
        eval_f1 += tmp_eval_f1
        nb_eval_steps += 1
        eval_balanced_accuracy += tmp_eval_balanced_accuracy

    f1_score = eval_f1/nb_eval_steps
    prec_score = eval_precision/nb_eval_steps
    recall_score = eval_recall/nb_eval_steps
    acc_score = eval_accuracy/nb_eval_steps

    balanced_acc_score = eval_balanced_accuracy/nb_eval_steps
    accuracy_metrics_dic = {
        "f1": f1_score,
        "Precision": prec_score,
        "Recall": recall_score,
        "acc": acc_score,
      "balance_acc": balanced_acc_score
    }

    print("F1 Score: ", f1_score)
    print("Precision Score: ", prec_score)
    print("Recall Score: ", recall_score)
    print("Acc Score: ", acc_score, "\n\n")

    # self.classifier.train()

    return (predictions_net, truths, identities_gender,identities_race, fine_terms,accuracy_metrics_dic)




  def train_eval(self, train_loader ,validation_loader,train_set,ippts_loader = None,fine_terms_list = None,iterations = 5,selection_score = "eq_odds",path ="", model_name = "",transformer_name = ""):

    self.fairness_metrics_iteration = []

    self.fairness_metrics_iteration_race = []
    self.fairness_metrics_iteration_gender = []
    self.performance_metrics_iteration = []

    self.fairness_metrics_iteration_race_fine = []
    self.fairness_metrics_iteration_gender_fine = []




    train_accs = []
    valid_accs = []

    best_fair_score = 0
    best_acc = 0
    gender_p_val = 0
    race_p_val = 0

    best_eq_score = 10000

    print("*"*10,"Start Debiasing Stage","*"*10, end = "\n")


    for iteration in range(iterations):  # loop over the dataset multiple times
      print("Iteration: ", iteration+1)

      start_time = time.time()
      self.train()
      end_time = time.time()

      train_iteration_duration = start_time - end_time
      print("train_iteration_duration",train_iteration_duration)
      print("\n")
      print('Validation metrics:')
      y_pred_val, actual_labels_val, protected_labels_gender,protected_labels_race, accuracy_metrics_dic_val = self.conduct_validation(validation_loader)
      valid_accs.append(accuracy_metrics_dic_val)
      accuracy_metrics_dic_val['MMC'] = matthews_corrcoef(actual_labels_val, y_pred_val)
      print("accuracy_metrics_dic_val",accuracy_metrics_dic_val)


      self.performance_metrics_iteration.append(accuracy_metrics_dic_val)





      print("Fairness Metrics on Validation:")

      metrics_dic_val_gender = metrics_eval.Metrics(y_pred_val,actual_labels_val, protected_labels_gender,thres).all_metrics()
      print("gender_val",metrics_dic_val_gender)

      metrics_dic_val_race = metrics_eval.Metrics(y_pred_val,actual_labels_val, protected_labels_race,thres).all_metrics()
      print("race_val",metrics_dic_val_race)

      # metrics_dic_val_gender['train_iteration_duration'] = train_iteration_duration
      # metrics_dic_val_gender['performance_metric'] =  accuracy_metrics_dic_val

      print("\n")

      if ippts_loader != None:

            print("Ippts Results")
            y_pred_ippts, actual_labels_ippts, protected_labels_gender_ippts,protected_labels_race_ippts,fine_terms_ippts ,accuracy_metrics_dic_ippts  = self.conduct_validation_ippts(ippts_loader)
            print("accuracy_metrics_dic_ippts",accuracy_metrics_dic_ippts)
            y_pred_ippts = y_pred_ippts - 1
            actual_labels_ippts = actual_labels_ippts - 1


            print("fine grain fairness results")
            fine_gender_metrics = metrics_eval.Metrics_fine_terms(y_pred_ippts,actual_labels_ippts, protected_labels_gender_ippts,fine_terms_ippts,fine_terms_list[0],thres).all_metrics_terms()
            print("fine_gender_metrics",fine_gender_metrics)
            fine_race_metrics = metrics_eval.Metrics_fine_terms(y_pred_ippts,actual_labels_ippts, protected_labels_race_ippts,fine_terms_ippts,fine_terms_list[1],thres).all_metrics_terms()
            print("fine_race_metrics",fine_race_metrics)
            self.fairness_metrics_iteration_race_fine.append(fine_race_metrics)
            self.fairness_metrics_iteration_gender_fine.append(fine_gender_metrics)

            print("\n")


      #--------------------------------------------------------------------------------
      self.fairness_metrics_iteration_race.append(metrics_dic_val_race)
      self.fairness_metrics_iteration_gender.append(metrics_dic_val_gender)


      #-------------------------------------------------------------------------------------------------------------------
      # define the fair_score properly:
      fair_score = accuracy_metrics_dic_val["acc"]

      # if best_acc < fair_score:
      #     best_acc = fair_score
      #     print("best_acc1:",best_acc)


      if selection_score == "eq_odds":

          # if (accuracy_metrics_dic_val["acc"] - best_acc >= 0.05) or ((abs(accuracy_metrics_dic_val["acc"] - best_acc)  <= 0.03) and (metrics_dic_val_gender['equ_odds']+metrics_dic_val_race['equ_odds'] < best_eq_score)):
          if best_acc < fair_score:


              best_acc = fair_score

              print("BEST fair score:",best_fair_score ,"new acc",accuracy_metrics_dic_val["acc"],"Epoch",iteration+1)
              print("best_acc:",best_acc)


              best_fair_score = fair_score
              best_eq_score =metrics_dic_val_gender['equ_odds']+metrics_dic_val_race['equ_odds']

              best_model_metrics ={}

              best_model_metrics['performance_metrics'] = accuracy_metrics_dic_val
              best_model_metrics['gender_metrics'] = metrics_dic_val_gender
              best_model_metrics['race_metrics'] = metrics_dic_val_race
              best_model_metrics["classifier"] = self.model.state_dict()
              best_model_metrics["EPOCH"] = iteration+1
              
              if ippts_loader != None:
                        best_model_metrics['gender_fine_grain'] = fine_gender_metrics
                        best_model_metrics["race_fine_grain"] = fine_race_metrics
              self.best_epoch = iteration

              self.save_best_model(best_model_metrics, path,model_name)
      else:

          # if  (accuracy_metrics_dic_val["acc"] - best_acc >= 0.05) or (abs(accuracy_metrics_dic_val["acc"] - best_acc)  <= 0.03) and (metrics_dic_val_gender['p_value']+metrics_dic_val_race['p_value'] > (gender_p_val +race_p_val)  ) :

              print("BEST fair score:",best_fair_score ,"new acc",accuracy_metrics_dic_val["acc"],"Epoch",iteration+1)

              best_fair_score = fair_score
              gender_p_val = metrics_dic_val_gender['p_value']
              race_p_val = metrics_dic_val_race['p_value']

              best_model_metrics ={}

              best_model_metrics['performance_metrics'] = accuracy_metrics_dic_val
              best_model_metrics['gender_metrics'] = metrics_dic_val_gender
              best_model_metrics['race_metrics'] = metrics_dic_val_race
              best_model_metrics["classifier"] = self.model.state_dict()
              best_model_metrics["EPOCH"] = iteration+1
                            
              if ippts_loader != None:
                        best_model_metrics['gender_fine_grain'] = fine_gender_metrics
                        best_model_metrics["race_fine_grain"] = fine_race_metrics
              self.best_epoch = iteration
              self.save_best_model(best_model_metrics, path,model_name)

    self.plot_iteration_score(self.performance_metrics_iteration,'acc')
    self.plot_iteration_score_mix([self.fairness_metrics_iteration_gender, self.fairness_metrics_iteration_race],'equ_odds',["gender","race"])
    self.plot_iteration_score_mix([self.fairness_metrics_iteration_gender, self.fairness_metrics_iteration_race],'p_value',["gender_p_value","race_p_value"])
    if ippts_loader != None:
        print("self.fairness_metrics_iteration_gender_fine",self.fairness_metrics_iteration_gender_fine)
        self.plot_iteration_score_fine_grain(self.fairness_metrics_iteration_gender_fine,"accuracy","acc",type_bias= "gender")
        self.plot_iteration_score_fine_grain(self.fairness_metrics_iteration_race_fine,"accuracy","acc",type_bias= "race")
        self.plot_iteration_score_fine_grain(self.fairness_metrics_iteration_gender_fine,"equ_odds",type_bias= "gender")
        self.plot_iteration_score_fine_grain(self.fairness_metrics_iteration_race_fine,"equ_odds",type_bias= "race")




    return self.performance_metrics_iteration, self.fairness_metrics_iteration_gender, self.fairness_metrics_iteration_race,best_model_metrics





  # set the path ----------------------------------------------------------
  def save_best_model(self,metrics, path,model_name):

      torch.save(metrics, path +model_name)

  def set_seed_val(self, seed_val):

      #------------------------------------------------------------------------------
      random.seed(seed_val)
      np.random.seed(seed_val)
      torch.manual_seed(seed_val)
      if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
      #------------------------------------------------------------------------------

  def plot_iteration_score(self,fairness_metrics_iteration, metric):


    metric_score_list = [ score[metric] for score in fairness_metrics_iteration ]
    plt.figure(100)

    plt.plot(self.best_epoch, metric_score_list[self.best_epoch], marker="*", markersize=10, markeredgecolor="red", markerfacecolor="green")


    plt.plot(metric_score_list, color='black', marker='o',mfc='pink' ) #plot the data
    plt.figure(100)

    plt.xticks(range(1,len(metric_score_list)+1, 1)) #set the tick frequency on x-axis

    plt.ylabel(metric) #set the label for y axis
    plt.xlabel('Epoch') #set the label for x-axis
    plt.title("Accuracy performance per epoch") #set the title of the graph
    plt.savefig(self.path + self.model_name.replace(".pt","") +'.png')
    plt.show() #display the graph

  def plot_iteration_score_mix(self,fairness_metrics_iterations, metric,legends = ["NO","s"]):

    color = ['black','red','orange']
    for idx,fairness_metrics_iteration in enumerate(fairness_metrics_iterations):
      metric_score_list = [ score[metric] for score in fairness_metrics_iteration ]

      plt.figure(100)

      plt.plot(metric_score_list, color=color[idx], marker='o',mfc='pink' ) #plot the data
      plt.plot(self.best_epoch, metric_score_list[self.best_epoch], marker="*", markersize=10, markeredgecolor="red", markerfacecolor="green")

      plt.figure(100)

      plt.xticks(range(0,len(metric_score_list)+1, 1)) #set the tick frequency on x-axis

      plt.ylabel(metric) #set the label for y axis
      plt.xlabel('Epochs') #set the label for x-axis
      plt.title("Fairness performance per epoch") #set the title of the graph


    plt.legend(legends)
    plt.savefig(self.path + self.model_name.replace(".pt","")+"_fairness"+metric +'.png')
    plt.show() #display the graph

  def freez_model_weights(self,model):
    for param in model.parameters():
      param.requires_grad = False

  def de_freez_model_weights(self, model):
    for param in model.parameters():
      param.requires_grad = True
