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

import sys
sys.path.append('/content/drive/MyDrive/Master_thesis/Metrics_Evaluations')
import metrics_eval


# address
import sys
sys.path.append('/content/drive/MyDrive/SS_Fair/Metrics_Evaluations')
import metrics_eval


# address
import sys
sys.path.append('/content/drive/MyDrive/SS_Fair/Models')
import Arch


import sys
sys.path.append('/content/drive/MyDrive/SS_Fair/Dataloaders/')
import data_loaders
import Data_sampler

import sys
sys.path.append('/content/drive/MyDrive/SS_Fair/Models/')
import Fair_NDABERT


noise_size = 100

#!pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
#!pip install sentencepiece
import pandas as pd
##Set random values

NUM_CLS = 3
# def set_seed_val(seed_val):

#     #------------------------------------------------------------------------------
#     seed_val = set_seed_val
#     random.seed(seed_val)
#     np.random.seed(seed_val)
#     torch.manual_seed(seed_val)
#     if torch.cuda.is_available():
#       torch.cuda.manual_seed_all(seed_val)
#     #------------------------------------------------------------------------------

epsilon = 1e-8
thres = 0.5


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




#------------------------------
#   The Generator as in
#   https://www.aclweb.org/anthology/2020.acl-main.191/
#   https://github.com/crux82/ganbert
#------------------------------

import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.ERROR)







class Training_eval(Fair_NDABERT.Pre_training_stage):

    def __init__(self, transformer,discriminator,generator,supervised_loss,discriminator_LR =  5e-5,generator_LR =  5e-5,batch_size = 32,path ="", model_name = "", transformer_name = ""):

      self.transformer = transformer
      self.discriminator = discriminator
      self.generator = generator

      transformer_vars = [i for i in self.transformer.parameters()]
      d_vars = transformer_vars + [v for v in self.discriminator.parameters()]
      g_vars = [v for v in self.generator.parameters()]

      self.dis_optimizer = torch.optim.AdamW(d_vars, lr=discriminator_LR)
      self.gen_optimizer = torch.optim.AdamW(g_vars, lr=generator_LR)

      self.supervised_loss = supervised_loss

      self.path = path
      self.model_name = model_name
      self.transformer_name = transformer_name
      self.batch_size = batch_size



    def train(self,train_iter,noise_size = 100):

      self.transformer.train()
      self.generator.train()
      self.discriminator.train()

      t0 = time.time()

      # Reset the total loss for this epoch.
      tr_g_loss = 0
      tr_d_loss = 0

      for step, batch in enumerate(train_iter):


          b_input_ids = batch['input_ids'].to(device)
          b_input_mask = batch['attention_mask'].to(device)
          b_labels = batch['targets'].to(device)
          b_label_mask = batch['label_mask'].to(device)

          gender_label = batch['identity_gender_term'].to(device)
          race_label = batch['identity_race_term'].to(device)
          weight = batch['weight'].to(device)



          model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
          hidden_states = model_outputs.last_hidden_state[:,0]

          NDA = hidden_states
          #add label stuff
          real_feature, real_logits, real_probs = self.discriminator(hidden_states)

          #---------------------------------
          # Simple Generator
          #---------------------------------

          #fake data the same batch size of unlabel data
          noise = Arch.noise_gen(b_input_ids.shape[0], noise_size, device)

          # Gnerate Fake data
          gen_rep = self.generator(noise)

          alpha = 0.9
          l = np.random.beta(alpha, alpha)
          l = max(l, 1-l)
          #l= 0.9
          l= 0.85
          neg_aug = l * gen_rep + (1 - l) * NDA
          neg_aug = neg_aug.to(device)


          fake_feature, fake_logits, fake_probs  = self.discriminator(neg_aug)






          #---------------------------------
          # Generator's LOSS estimation
          #---------------------------------
          g_loss_d =  -1 * torch.mean(torch.log(1 - fake_probs[:, -1] + epsilon))

          g_feat_reg = torch.mean(torch.pow(torch.mean(real_feature, dim=0) - torch.mean(fake_feature, dim=0), 2))
          g_loss = g_loss_d + g_feat_reg




          #---------------------------------
          # Discriminator's LOSS estimation
          #---------------------------------
          D_L_Supervised = self.supervised_loss(real_logits,b_labels,b_label_mask,weight)
          # D_L_unsupervised1U = -1 * torch.mean(torch.log(1 - weight*real_probs[:, -1] + epsilon))
          D_L_unsupervised1U = -1 * torch.mean(torch.log(1 - weight*real_probs[:, -1] + epsilon))

          D_L_unsupervised2U = -1 * torch.mean(torch.log(fake_probs[:, -1] + epsilon))
          d_loss = D_L_Supervised + D_L_unsupervised1U + D_L_unsupervised2U


          #---------------------------------
          #  OPTIMIZATION
          #---------------------------------
          # Avoid gradient accumulation
          self.gen_optimizer.zero_grad()
          self.dis_optimizer.zero_grad()

          # Calculate weigth updates
          # retain_graph=True is required since the underlying graph will be deleted after backward
          g_loss.backward(retain_graph=True)
          # d_loss.backward()
          d_loss.backward()

          # Apply modifications
          self.gen_optimizer.step()
          self.dis_optimizer.step()

          # Save the losses to print them later
          tr_g_loss += g_loss.item()
          tr_d_loss += d_loss.item()


      # Calculate the average loss over all of the batches.
      ###### it is very important how many times we use label data and how we calculate the loss
      avg_train_loss_g = tr_g_loss / len(train_iter)
      avg_train_loss_d = tr_d_loss/ len(train_iter)

      # Measure how long this epoch took.

      print("")
      print("  Average training loss generetor: {0:.3f}".format(avg_train_loss_g))
      print("  Average training loss discriminator: {0:.3f}".format(avg_train_loss_d))

      result_dic = {

          'Training Loss generator': avg_train_loss_g,
          'Training Loss discriminator sup': avg_train_loss_d,
      }

      return result_dic

    def conduct_validation(self,data_loader):

      print("Running Test...")

      self.transformer.eval() #maybe redundant
      self.discriminator.eval()
      self.generator.eval()

      #-------------------------------------------
      eval_loss, eval_balanced_accuracy, eval_accuracy, eval_precision, eval_recall, eval_f1 = 0,0, 0, 0, 0, 0
      nb_eval_steps = 0

      predictions_net = np.empty((0,))
      truths = np.empty((0,))
      identities_gender = np.empty((0,))
      identities_race = np.empty((0,))

      correct_net = 0
      total = 0
      #--------------------------------------------------


      # Tracking variables
      total_test_accuracy = 0
      total_test_loss = 0
      nb_test_steps = 0


      # Evaluate data for one epoch
      for batch in data_loader:

          # Unpack this training batch from our dataloader.
          b_input_ids = batch['input_ids'].to(device)
          b_input_mask = batch['attention_mask'].to(device)
          b_labels = batch['targets'].to(device)
          b_label_mask = batch['label_mask'].to(device)
          gender_label = batch['identity_gender_term'].to(device)
          race_label = batch['identity_race_term'].to(device)


          # Tell pytorch not to bother with constructing the compute graph during
          # the forward pass, since this is only needed for backprop (training).
          with torch.no_grad():

              model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
              hidden_states = model_outputs.last_hidden_state[:,0]

              _, logits, probs = self.discriminator(hidden_states)
              filtered_logits = logits[:,0:-1]



          # Accumulate the predictions and the input labels
          _, preds = torch.max(filtered_logits, 1)
          #------------------------------------
          batch_size = b_labels.size(0)
          total += batch_size
          correct_net_batch = (preds == b_labels).sum().item()
          correct_net += correct_net_batch

          pred = preds.detach().cpu().numpy()
          label_ids = b_labels.to('cpu').numpy()

          predictions_net = np.concatenate((predictions_net, pred))
          truths = np.concatenate((truths, label_ids))
          identities_gender = np.concatenate((identities_gender, gender_label.cpu().numpy()))
          identities_race = np.concatenate((identities_race, race_label.cpu().numpy()))



          #compute the performance
          tmp_eval_balanced_accuracy,tmp_eval_accuracy, tmp_eval_precision, temp_eval_recall, tmp_eval_f1 = metrics_eval.Performance_Metrics.get_metrics(label_ids, pred)

          eval_accuracy += tmp_eval_accuracy
          eval_precision += tmp_eval_precision
          eval_recall += temp_eval_recall
          eval_f1 += tmp_eval_f1
          nb_eval_steps += 1
          eval_balanced_accuracy += tmp_eval_balanced_accuracy

          #------------------------------------


      #---------------------------------------------------
      f1_score = eval_f1/nb_eval_steps
      prec_score = eval_precision/nb_eval_steps
      recall_score = eval_recall/nb_eval_steps
      acc_score = eval_accuracy/nb_eval_steps
      balanced_acc_score = eval_balanced_accuracy/nb_eval_steps

      print("acc test: ", metrics_eval.Performance_Metrics.get_metrics(truths, predictions_net))

      accuracy_metrics_dic = {
            "f1": f1_score,
            "Precision": prec_score,
            "Recall": recall_score,
            "acc": acc_score,
        "balance_acc": balanced_acc_score
      }
      return (predictions_net, truths, identities_gender,identities_race, accuracy_metrics_dic)


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
      self.discriminator.eval()
      self.transformer.eval()
      with torch.no_grad(): # IMPORTANT: we don't want to do back prop during validation/testing!

        for batch in data_loader:

          # Unpack this training batch from our dataloader.
          b_input_ids = batch['input_ids'].to(device)
          b_input_mask = batch['attention_mask'].to(device)
          targets = batch['targets'].to(device)

          identity_term_gender = batch['identity_gender_term'].to(device)
          identity_term_race = batch['identity_race_term'].to(device)
          fine_term = batch['attr']

          model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
          hidden_states = model_outputs.last_hidden_state[:,0]
          classifier_prev_output, real_logits, real_probs = self.discriminator(hidden_states)
          filtered_logits = real_logits[:,0:-1]

          _, net_predicted = torch.max(filtered_logits, 1)


          # net_outputs, net_prev_outputs = self.classifier(input_ids,attention_mask)
          # _, net_predicted = torch.max(net_outputs.data, 1)

          batch_size = targets.size(0)
          total += batch_size
          correct_net_batch = (net_predicted == targets).sum().item()
          correct_net += correct_net_batch


          predictions_net = np.concatenate((predictions_net, net_predicted.cpu().numpy()))
          truths = np.concatenate((truths, targets.cpu().numpy()))
          identities_gender = np.concatenate((identities_gender, identity_term_gender.cpu().numpy()))
          identities_race = np.concatenate((identities_race, identity_term_race.cpu().numpy()))
          fine_terms += fine_term



          pred = net_predicted.detach().cpu().numpy()
          label_ids = targets.to('cpu').numpy()

          label_ids1 = label_ids -1
          pred1 = pred -1

          tmp_eval_balanced_accuracy,tmp_eval_accuracy, tmp_eval_precision, temp_eval_recall, tmp_eval_f1 = metrics_eval.Performance_Metrics.get_metrics(label_ids1, pred1)

          eval_accuracy += tmp_eval_accuracy
          eval_precision += tmp_eval_precision
          eval_recall += temp_eval_recall
          eval_f1 += tmp_eval_f1
          nb_eval_steps += 1

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
      print("Balanced Acc Score: ", balanced_acc_score)

      print("Acc Score: ", acc_score, "\n\n")

      # self.classifier.train()

      return (predictions_net, truths, identities_gender,identities_race, fine_terms,accuracy_metrics_dic)

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
      self.discriminator.eval()
      self.transformer.eval()
      with torch.no_grad(): # IMPORTANT: we don't want to do back prop during validation/testing!

        for batch in data_loader:

          # Unpack this training batch from our dataloader.
          b_input_ids = batch['input_ids'].to(device)
          b_input_mask = batch['attention_mask'].to(device)
          targets = batch['targets'].to(device)

          identity_term_gender = batch['identity_gender_term'].to(device)
          identity_term_race = batch['identity_race_term'].to(device)
          fine_term = batch['attr']

          model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
          hidden_states = model_outputs.last_hidden_state[:,0]
          classifier_prev_output, real_logits, real_probs = self.discriminator(hidden_states)
          filtered_logits = real_logits[:,0:-1]

          _, net_predicted = torch.max(filtered_logits, 1)


          # net_outputs, net_prev_outputs = self.classifier(input_ids,attention_mask)
          # _, net_predicted = torch.max(net_outputs.data, 1)

          batch_size = targets.size(0)
          total += batch_size
          correct_net_batch = (net_predicted == targets).sum().item()
          correct_net += correct_net_batch


          predictions_net = np.concatenate((predictions_net, net_predicted.cpu().numpy()))
          truths = np.concatenate((truths, targets.cpu().numpy()))
          identities_gender = np.concatenate((identities_gender, identity_term_gender.cpu().numpy()))
          identities_race = np.concatenate((identities_race, identity_term_race.cpu().numpy()))
          fine_terms += fine_term



          pred = net_predicted.detach().cpu().numpy()
          label_ids = targets.to('cpu').numpy()

          label_ids1 = label_ids -1
          pred1 = pred -1

          tmp_eval_balanced_accuracy, tmp_eval_accuracy, tmp_eval_precision, temp_eval_recall, tmp_eval_f1 = metrics_eval.Performance_Metrics.get_metrics(label_ids1, pred1)

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
      print("Balanced Acc Score: ", balanced_acc_score)

      print("Acc Score: ", acc_score, "\n\n")

      # self.classifier.train()

      return (predictions_net, truths, identities_gender,identities_race, fine_terms,accuracy_metrics_dic)



    # None,gender_idnetity_loader,gender_idnetity_loader_val,gender_dataset, 1,1,1
    def train_eval(self, train_loader ,validation_loader,train_set,ippts_loader = None,fine_terms_list = None, iterations = 15,noise_size =100, selection_score = "eq_odds",path ="", model_name = "",transformer_name = ""):



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
        print("*"*10,"Epoch","*"*10,iteration+1 )

        start_time = time.time()
        self.train(train_loader,noise_size = 100)

        end_time = time.time()

        train_iteration_duration = start_time - end_time
        # print("train_iteration_duration",train_iteration_duration)

        print('Validation metrics:')
        y_pred_val, actual_labels_val, protected_labels_gender,protected_labels_race, accuracy_metrics_dic_val = self.conduct_validation(validation_loader)
        valid_accs.append(accuracy_metrics_dic_val)
        accuracy_metrics_dic_val['MMC'] = matthews_corrcoef(actual_labels_val, y_pred_val)
        print("accuracy_metrics_dic_val",accuracy_metrics_dic_val)

        self.performance_metrics_iteration.append(accuracy_metrics_dic_val)

        y_pred_val = y_pred_val - 1
        actual_labels_val = actual_labels_val - 1



        print("Fairness Metrics on Validation:")

        metrics_dic_val_gender = metrics_eval.Metrics(y_pred_val,actual_labels_val, protected_labels_gender,thres).all_metrics()
        print("gender_val",metrics_dic_val_gender)

        metrics_dic_val_race = metrics_eval.Metrics(y_pred_val,actual_labels_val, protected_labels_race,thres).all_metrics()
        print("race_val",metrics_dic_val_race)

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

        if selection_score == "eq_odds":
          
          # if (accuracy_metrics_dic_val["acc"] - best_acc >= 0.05) or ((abs(accuracy_metrics_dic_val["acc"] - best_acc)  <= 0.03) and (metrics_dic_val_gender['equ_odds']+metrics_dic_val_race['equ_odds'] < best_eq_score)):  
          if best_acc < fair_score:
                print("BEST fair score:",best_fair_score ,"new acc",accuracy_metrics_dic_val["acc"],"Epoch",iteration+1)

                best_acc = fair_score
                best_fair_score = fair_score
                best_eq_score =metrics_dic_val_gender['equ_odds']+metrics_dic_val_race['equ_odds']

                best_model_metrics ={}

                best_model_metrics['performance_metrics'] = accuracy_metrics_dic_val
                best_model_metrics['gender_metrics'] = metrics_dic_val_gender
                best_model_metrics['race_metrics'] = metrics_dic_val_race
                best_model_metrics["discriminator"] = self.discriminator.state_dict()
                best_model_metrics["EPOCH"] = iteration+1
                                
                if ippts_loader != None:
                          best_model_metrics['gender_fine_grain'] = fine_gender_metrics
                          best_model_metrics["race_fine_grain"] = fine_race_metrics
                self.best_epoch = iteration

                self.save_best_model(best_model_metrics, path,model_name)
                self.transformer.save_pretrained(path+transformer_name)
        else:

            if  (accuracy_metrics_dic_val["acc"] - best_acc >= 0.05) or (abs(accuracy_metrics_dic_val["acc"] - best_acc)  <= 0.03) and (metrics_dic_val_gender['p_value']+metrics_dic_val_race['p_value'] > (gender_p_val +race_p_val)  ) :


                print("BEST fair score:",best_fair_score ,"new acc",accuracy_metrics_dic_val["acc"],"Epoch",iteration+1)

                best_fair_score = fair_score
                gender_p_val = metrics_dic_val_gender['p_value']
                race_p_val = metrics_dic_val_race['p_value']

                best_model_metrics ={}

                best_model_metrics['performance_metrics'] = accuracy_metrics_dic_val
                best_model_metrics['gender_metrics'] = metrics_dic_val_gender
                best_model_metrics['race_metrics'] = metrics_dic_val_race
                best_model_metrics["discriminator"] = self.discriminator.state_dict()
                best_model_metrics["EPOCH"] = iteration+1                
                if ippts_loader != None:
                          best_model_metrics['gender_fine_grain'] = fine_gender_metrics
                          best_model_metrics["race_fine_grain"] = fine_race_metrics
                self.best_epoch = iteration
                self.save_best_model(best_model_metrics, path,model_name)
                self.transformer.save_pretrained(path+transformer_name)

      if ippts_loader != None:
        print("self.fairness_metrics_iteration_gender_fine",self.fairness_metrics_iteration_gender_fine)
        self.plot_iteration_score_fine_grain(self.fairness_metrics_iteration_gender_fine,"accuracy","acc",type_bias= "gender")
        self.plot_iteration_score_fine_grain(self.fairness_metrics_iteration_race_fine,"accuracy","acc",type_bias= "race")
        self.plot_iteration_score_fine_grain(self.fairness_metrics_iteration_gender_fine,"equ_odds",type_bias= "gender")
        self.plot_iteration_score_fine_grain(self.fairness_metrics_iteration_race_fine,"equ_odds",type_bias= "race")


      return self.performance_metrics_iteration, self.fairness_metrics_iteration_gender, self.fairness_metrics_iteration_race, best_model_metrics


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
