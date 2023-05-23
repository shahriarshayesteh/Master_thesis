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

import pandas as pd


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


import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.ERROR)


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
from Arch import BertClassifier


import sys
sys.path.append('/content/drive/MyDrive/SS_Fair/Dataloaders/')
import data_loaders
import Data_sampler

import sys
sys.path.append('/content/drive/MyDrive/SS_Fair/Models/')
import Fair_NDABERT

from transformers import DistilBertTokenizer, DistilBertModel

thres = 0.5



# address
import sys
sys.path.append('/content/drive/MyDrive/SS_Fair/Models')
from  Arch import noise_gen


import sys
sys.path.append('/content/drive/MyDrive/SS_Fair/Models')
from  Parent_Models import TextClassificationModel, DebiasingMethod, TrainingLog, PlottingUtils


# BERT class that inherits from TextClassificationModel
class BERT(TextClassificationModel):

    def __init__(self, classifier, adversary,supervised_loss, loss_adversary, log_training,plot_training,seed,classifier_LR=2e-5, adversary_LR=0.001, lda=[15, 30],batch_size = 32, path="", model_name="", transformer_name=""):

        super().__init__(log_training,plot_training)

        self.classifier = classifier
        self.adversary = adversary

        self.classifier_LR = classifier_LR
        self.adversary_LR = adversary_LR

        self.optimizer_classifier = torch.optim.AdamW(self.classifier.parameters(), lr= self.classifier_LR)
        self.optimizer_adversary  = torch.optim.AdamW(self.adversary.parameters(), lr=self.adversary_LR)

        self.loss_classifier = supervised_loss
        self.loss_adversary = loss_adversary


        self.lda = lda
        self.batch_size = batch_size
        self.path = path
        self.model_name = model_name
        self.transformer_name = transformer_name
        self.log_training = log_training
        self.plot_training = plot_training

        self.seed= seed
        self.set_seed_val(self.seed)


    def pretrain_classifier(self,train_loader,epochs):

      self.freez_model_weights(self.adversary)
      self.classifier.train()
      pretrain_classifier_loss = 0
      steps = 0

      for epoch in range(epochs):

        print("Epoch: ", epoch + 1)
        epoch_loss = 0
        epoch_batches = 0
        for step, batch in enumerate(train_loader):


            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['targets'].to(device)
            b_label_mask = batch['label_mask'].to(device)
            gender_label = batch['identity_gender_term'].to(device)
            race_label = batch['identity_race_term'].to(device)


            self.optimizer_classifier.zero_grad()

            classifier_output,logit, prob = self.classifier(b_input_ids,b_input_mask)
            confidence, predicted_label = torch.max(prob, 1)

            classifier_loss = self.loss_classifier(logit, b_labels) # compute loss
            classifier_loss.backward() # back prop
            self.optimizer_classifier.step()
            pretrain_classifier_loss += classifier_loss.item()
            epoch_loss += classifier_loss.item()
            epoch_batches += 1
            steps += 1

        # print("Average Pretrain Classifier epoch loss: ", epoch_loss/epoch_batches)
      print("Average Pretrain Classifier batch loss: ", pretrain_classifier_loss/steps)

      self.de_freez_model_weights(self.adversary)
      return self.classifier

    def pretrain_adversary(self,train_loader,epochs =1):

      self.freez_model_weights(self.classifier)
      self.adversary.train()
      pretrain_adversary_loss = 0
      steps = 0

      for epoch in range(epochs):

        print("Epoch: ", epoch + 1)
        epoch_loss = 0
        epoch_batches = 0
        for step, batch in enumerate(train_loader):


            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            b_label_mask = batch['label_mask'].to(device)
            identity_term_gender = batch['identity_gender_term'].to(device)
            identity_term_race = batch['identity_race_term'].to(device)

            self.optimizer_adversary.zero_grad()

            classifier_prev_output,logit, prob = self.classifier(input_ids,attention_mask)
            confidence, predicted_label = torch.max(prob, 1)

            adversary_output_gender,adversary_output_race = self.adversary(classifier_prev_output)
            adversary_loss_gender = self.loss_adversary(adversary_output_gender, identity_term_gender) # compute loss

            #CHANGE
            adversary_loss_race = self.loss_adversary(adversary_output_race, identity_term_race) # compute loss
            adversary_loss = self.lda[0]*adversary_loss_gender + self.lda[1]*adversary_loss_race          #adversary_loss = adversary_loss_gender

            adversary_loss.backward() # back prop
            self.optimizer_adversary.step()
            pretrain_adversary_loss += adversary_loss.item()
            epoch_loss += adversary_loss.item()
            epoch_batches += 1
            steps += 1

        print("Average Pretrain Adversary epoch loss: ", epoch_loss/epoch_batches)
      print("Average Pretrain Adversary batch loss: ", pretrain_adversary_loss/steps)

      self.de_freez_model_weights(self.classifier)
      return  self.adversary

    def conduct_validation(self, data_loader,epoch, mode = "Testing"):

      """
      Conducts validation on the model by running it on the validation dataset.
      Calculates various performance metrics on the coarse term and logs them.

      Parameters:
      data_loader (DataLoader): The data loader for the validation dataset.
      epoch (int): The current epoch number.
      mode (str): The mode of validation, either "Testing" or "Validation".

      Returns:
      None
      """

      print("Running Test...")

      self.classifier.eval()
      self.adversary.eval()

      data_dict = {"predictions_net": [], "truths": [], "identities_gender": [], "identities_race": [], "y_scores": []}

      for batch in data_loader:
          # Unpack this training batch from our dataloader.
          b_input_ids = batch['input_ids'].to(device)
          b_input_mask = batch['attention_mask'].to(device)
          b_labels = batch['targets'].to(device)
          gender_label = batch['identity_gender_term'].to(device)
          race_label = batch['identity_race_term'].to(device)

          # Tell pytorch not to bother with constructing the compute graph during
          # the forward pass, since this is only needed for backprop (training).
          with torch.no_grad():
              #------
              classifier_prev_output,logits, probs = self.classifier(b_input_ids,b_input_mask)




          _, preds = torch.max(probs, 1)


          '''
          0,1 coverter
          '''
          # This will map all the 2s to 1s and all other values to 0s.
          label_ids = b_labels.cpu().numpy()
          pred = preds.detach().cpu().numpy()
          gender_label = gender_label.cpu().numpy()
          race_label = race_label.cpu().numpy()
          y_score = probs[:,1:2]

          data_dict["predictions_net"].extend(pred)
          data_dict["truths"].extend(label_ids)
          data_dict["identities_gender"].extend(gender_label)
          data_dict["identities_race"].extend(race_label)
          data_dict["y_scores"].extend(y_score.cpu().numpy())

      return self.performance_metrics(data_dict, epoch,mode)


    def conduct_validation_fine_term(self,data_loader,epoch,fine_terms_list ,mode = "Testing"):

      """
      Conducts validation on the model by running it on the validation dataset.
      Calculates various performance metrics on the fine terms and logs them.

      Parameters:
      data_loader (DataLoader): The data loader for the validation dataset.
      epoch (int): The current epoch number.
      mode (str): The mode of validation, either "Testing" or "Validation".

      Returns:
      None
      """

      print("Running Test...")

      self.classifier.eval()
      self.adversary.eval()

      data_dict = {"predictions_net": [], "truths": [], "identities_gender": [], "identities_race": [], "y_scores": [], "fine_terms": []}

      for batch in data_loader:

          # Unpack this training batch from our dataloader.
          b_input_ids = batch['input_ids'].to(device)
          b_input_mask = batch['attention_mask'].to(device)
          b_labels = batch['targets'].to(device)
          gender_label = batch['identity_gender_term'].to(device)
          race_label = batch['identity_race_term'].to(device)
          fine_term = batch['attr']

          # Tell pytorch not to bother with constructing the compute graph during
          # the forward pass, since this is only needed for backprop (training).

          with torch.no_grad():
              #------
              classifier_prev_output,logits, probs = self.classifier(b_input_ids,b_input_mask)




          _, preds = torch.max(probs, 1)


          '''
          0,1 coverter
          '''
          # This will map all the 2s to 1s and all other values to 0s.
          label_ids = b_labels.cpu().numpy()
          pred = preds.detach().cpu().numpy()
          gender_label = gender_label.cpu().numpy()
          race_label = race_label.cpu().numpy()
          y_score = probs[:,1:2]


          data_dict["predictions_net"].extend(pred)
          data_dict["truths"].extend(label_ids)
          data_dict["identities_gender"].extend(gender_label)
          data_dict["identities_race"].extend(race_label)
          data_dict["y_scores"].extend(y_score.cpu().numpy())
          data_dict["fine_terms"].extend(fine_term)

      # self.performance_metrics_fine_term(data_dict)
      return self.performance_metrics(data_dict, epoch,mode), self.performance_metrics_fine_term(data_dict, epoch,fine_terms_list,mode)

    def pre_training(self,train_loader,validation_loader,mode= "Validation", epochs_adv = 1, iterations = 1,noise_size = 100, l = 0.85,ippts_loader= None,fine_terms_list = None,path ="", model_name = "",transformer_name = ""):

        valid_accs = []
        best_fair_score = 0


        self.fairness_metrics_iteration_race = []
        self.fairness_metrics_iteration_gender = []
        self.performance_metrics_iteration = []

        self.fairness_metrics_iteration_race_fine = []
        self.fairness_metrics_iteration_gender_fine = []
        performance_log = {}


        for iteration in range(iterations):

          start_time= time.time()

          print("Pre-Train classifier", end = '\n')
          print("DDDDDDDD", next(iter(train_loader)))
          self.classifier = self.pretrain_classifier(train_loader,1)

          # self.pretrain_classifier(train_loader,l,noise_size)

          print('Validation metrics:')
          # performance_log = self.conduct_validation(validation_loader,iteration)

          print("Train Adversay", end = '\n')
          self.adversary = self.pretrain_adversary(train_loader,epochs_adv)

          print("\n")

          if ippts_loader != None:

                iteration = iteration
                print("Fine_term Results")
                performance_log,performance_log_fine_terms  = self.conduct_validation_fine_term(ippts_loader,iteration,fine_terms_list)
                print("performance_log1",performance_log)
                



                end_time= time.time()

                if performance_log['Accuracy'] > best_fair_score:

                  print("BEST fair score:",best_fair_score ,"new acc",performance_log["Accuracy"],"Epoch",iteration+1)

                  best_fair_score = performance_log['Accuracy']

                  best_model_metrics ={}
                  self.best_epoch = iteration
                  best_model_metrics['performance_metrics'] = performance_log
                  best_model_metrics["classifier"] = self.classifier.state_dict()
                  best_model_metrics["adversary"] = self.adversary.state_dict()
                  # best_model_metrics["generator"] = self.generator.state_dict()


                  print("performance_log",performance_log)

                  best_model_metrics["EPOCH"] = iteration+1

                  self.save_best_model(best_model_metrics)
                  #self.classifier.save_pretrained(self.path+self.transformer_name)

                  best_perofrmance_log = performance_log
                  if ippts_loader != None:
                      best_perofrmance_log_fine = performance_log_fine_terms

        #----self.path is for NDABERT but we want path to fairNDABERT is given as path-------------------------------------------------------------------------------------------
        self.log_training.save_log(self.path)
        if ippts_loader != None:
          self.log_training.save_log(self.path,True)

        self.log_training.save_best_log(self.path,best_perofrmance_log,fine_term= False)
        if ippts_loader != None:
          self.log_training.save_best_log(self.path,best_perofrmance_log_fine,fine_term= True)

        epoch_coarse = self.log_training.get_epoch_coarse()
        if ippts_loader != None:
          epoch_fine = self.log_training.get_epoch_fine()


        self.plot_training.plot_iteration_score(epoch_coarse,["Accuracy","race_equ_odds_percent","gender_equ_odds_percent","race_p_value","gender_p_value"],best_model_metrics["EPOCH"])
        self.plot_training.plot_iteration_score(epoch_coarse,["race_equ_odds","gender_equ_odds"],best_model_metrics["EPOCH"])

        if ippts_loader != None:

          self.plot_training.plot_iteration_score_fine_grain(epoch_fine, "equ_odds_percent", best_model_metrics["EPOCH"],"gender")
          self.plot_training.plot_iteration_score_fine_grain(epoch_fine, "equ_odds_percent", best_model_metrics["EPOCH"],"race")



        return best_perofrmance_log, best_perofrmance_log_fine,epoch_coarse,epoch_fine



    def set_seed_val(self, seed_val):

      #------------------------------------------------------------------------------
      random.seed(seed_val)
      np.random.seed(seed_val)
      torch.manual_seed(seed_val)
      if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
      #------------------------------------------------------------------------------
    def freez_model_weights(self,model):
      for param in model.parameters():
        param.requires_grad = False

    def de_freez_model_weights(self, model):
      for param in model.parameters():
        param.requires_grad = True

    # set the path ----------------------------------------------------------
    def save_best_model(self,metrics):

      torch.save(metrics, self.path +self.model_name)



class FairBERT(DebiasingMethod):

    def __init__(self, adversary,loss_classifier,loss_adversary,log_training,plot_training,seed,adversary_LR = 0.001,classifier_LR =1e-5,lda= [1,1],batch_size = 8,path ="", model_name = "",transformer_name = ""):


      ############
      checkpoint = torch.load(path+model_name)

      model_FB = BertClassifier(2)
      model_FB.load_state_dict(checkpoint['classifier'])

      adversary.load_state_dict(checkpoint['adversary'])
      model_FB = model_FB.to(device)

      adversary.to(device)

      #####

      self.adversary = adversary
      self.classifier = model_FB


      self.optimizer_classifier = torch.optim.AdamW(self.classifier.parameters(), lr=classifier_LR)
      self.optimizer_adversary  = torch.optim.Adam(self.adversary.parameters(), lr=adversary_LR)

      self.loss_classifier = loss_classifier
      self.loss_adversary = loss_adversary
      self.path = path
      self.model_name = model_name
      self.transformer_name = transformer_name
      self.log_training = log_training
      self.plot_training = plot_training

      self.lda = lda
      self.batch_size = batch_size

    def conduct_validation(self, data_loader,epoch, mode = "Testing", save = True):

      """
      Conducts validation on the model by running it on the validation dataset.
      Calculates various performance metrics on the coarse term and logs them.

      Parameters:
      data_loader (DataLoader): The data loader for the validation dataset.
      epoch (int): The current epoch number.
      mode (str): The mode of validation, either "Testing" or "Validation".

      Returns:
      None
      """

      print("Running Test...")

      self.classifier.eval()
      self.adversary.eval()

      data_dict = {"predictions_net": [], "truths": [], "identities_gender": [], "identities_race": [], "y_scores": []}

      for batch in data_loader:
          # Unpack this training batch from our dataloader.
          b_input_ids = batch['input_ids'].to(device)
          b_input_mask = batch['attention_mask'].to(device)
          b_labels = batch['targets'].to(device)
          gender_label = batch['identity_gender_term'].to(device)
          race_label = batch['identity_race_term'].to(device)

          # Tell pytorch not to bother with constructing the compute graph during
          # the forward pass, since this is only needed for backprop (training).
          with torch.no_grad():
              #------
              classifier_prev_output,logits, probs = self.classifier(b_input_ids,b_input_mask)




          _, preds = torch.max(probs, 1)


          '''
          0,1 coverter
          '''
          # This will map all the 2s to 1s and all other values to 0s.
          label_ids = b_labels.cpu().numpy()
          pred = preds.detach().cpu().numpy()
          gender_label = gender_label.cpu().numpy()
          race_label = race_label.cpu().numpy()
          y_score = probs[:,1:2]

          data_dict["predictions_net"].extend(pred)
          data_dict["truths"].extend(label_ids)
          data_dict["identities_gender"].extend(gender_label)
          data_dict["identities_race"].extend(race_label)
          data_dict["y_scores"].extend(y_score.cpu().numpy())

      return self.performance_metrics(data_dict, epoch,mode,save)


    def conduct_validation_fine_term(self,data_loader,epoch, fine_terms_list,save = True, mode = "Testing"):

      """
      Conducts validation on the model by running it on the validation dataset.
      Calculates various performance metrics on the fine terms and logs them.

      Parameters:
      data_loader (DataLoader): The data loader for the validation dataset.
      epoch (int): The current epoch number.
      mode (str): The mode of validation, either "Testing" or "Validation".

      Returns:
      None
      """

      print("Running Test...")

      self.classifier.eval()
      self.adversary.eval()

      data_dict = {"predictions_net": [], "truths": [], "identities_gender": [], "identities_race": [], "y_scores": [], "fine_terms": []}

      for batch in data_loader:

          # Unpack this training batch from our dataloader.
          b_input_ids = batch['input_ids'].to(device)
          b_input_mask = batch['attention_mask'].to(device)
          b_labels = batch['targets'].to(device)
          gender_label = batch['identity_gender_term'].to(device)
          race_label = batch['identity_race_term'].to(device)
          fine_term = batch['attr']

          # Tell pytorch not to bother with constructing the compute graph during
          # the forward pass, since this is only needed for backprop (training).

          with torch.no_grad():
              #------
              classifier_prev_output,logits, probs = self.classifier(b_input_ids,b_input_mask)




          _, preds = torch.max(probs, 1)


          '''
          0,1 coverter
          '''
          # This will map all the 2s to 1s and all other values to 0s.
          label_ids = b_labels.cpu().numpy()
          pred = preds.detach().cpu().numpy()
          gender_label = gender_label.cpu().numpy()
          race_label = race_label.cpu().numpy()
          y_score = probs[:,1:2]


          data_dict["predictions_net"].extend(pred)
          data_dict["truths"].extend(label_ids)
          data_dict["identities_gender"].extend(gender_label)
          data_dict["identities_race"].extend(race_label)
          data_dict["y_scores"].extend(y_score.cpu().numpy())
          data_dict["fine_terms"].extend(fine_term)

      # self.performance_metrics_fine_term(data_dict)
      return self.performance_metrics(data_dict, epoch,mode,save), self.performance_metrics_fine_term(data_dict, epoch,fine_terms_list,mode)






    def train_adversary(self,train_loader, epochs=1):

      adv_loss = 0
      steps = 0
      self.adversary.train()

      # self.freez_model_weights(self.transformer)
      self.freez_model_weights(self.classifier)

      for epoch in range(epochs):

        for step, batch in enumerate(train_loader):

            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            b_label_mask = batch['label_mask'].to(device)
            identity_term_gender = batch['identity_gender_term'].to(device)
            identity_term_race = batch['identity_race_term'].to(device)


            self.loss_adversary.zero_grad()

            classifier_prev_output,logit, prob = self.classifier(b_input_ids,b_input_mask)
            confidence, predicted_label = torch.max(prob, 1)

            adversary_output_gender,adversary_output_race = self.adversary(classifier_prev_output)
            adversary_loss_gender = self.loss_adversary(adversary_output_gender, identity_term_gender) # compute loss
            adversary_loss_race = self.loss_adversary(adversary_output_race, identity_term_race) # compute loss
            adversary_loss = adversary_loss_gender + adversary_loss_race

            self.loss_adversary.zero_grad()
            adversary_loss.backward() # back prop
            self.optimizer_adversary.step()
            #-------------------------------------------------------------------------------
            adv_loss += adversary_loss.item()
            steps += 1

      print("Average Adversary batch loss: ", adv_loss/steps)

      # self.de_freez_model_weights(self.transformer)
      self.de_freez_model_weights(self.classifier)

    def train_classifier_rnd_batch(self,train_set, num_rnd_batch =2,noise_size = 100):

      self.freez_model_weights(self.adversary)
      self.classifier.train()
      pretrain_classifier_loss = 0
      steps = 0
      rnd_batch_gender_loader = data_loaders.create_rand_batch_Fair_data_loader(train_set ,self.batch_size, num_rnd_batch)
      iter_data = iter(rnd_batch_gender_loader)
      flag = True
      # while flag:
      try:

          # Loop through the data loader
          for step, batch in enumerate(rnd_batch_gender_loader):
            print("step",step,steps)
            steps +=1

            # Load data to device
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['targets'].to(device)
            b_label_mask = batch['label_mask'].to(device)
            identity_term_gender = batch['identity_gender_term'].to(device)
            identity_term_race = batch['identity_race_term'].to(device)

            self.optimizer_classifier.zero_grad()

            classifier_prev_output,logit, prob = self.classifier(b_input_ids,b_input_mask)
            confidence, predicted_label = torch.max(prob, 1)

            adversary_output_gender,adversary_output_race = self.adversary(classifier_prev_output)
            adversary_loss_gender = self.loss_adversary(adversary_output_gender, identity_term_gender) # compute loss
            #CHANGE
            adversary_loss_race = self.loss_adversary(adversary_output_race, identity_term_race) # compute loss
            adversary_loss = self.lda[0]*adversary_loss_gender + self.lda[1]*adversary_loss_race
            #adversary_loss = adversary_loss_gender
            # print("LLLLLLLLLLL",self.lda[0],self.lda[1])


            classifier_loss = self.loss_classifier(logit, b_labels) # compute loss
            total_classifier_loss = classifier_loss - adversary_loss
            print("total_classifier_loss",total_classifier_loss)
            total_classifier_loss.backward() # back prop

            self.optimizer_classifier.step()

            print("Adversary {} Mini-Batch loss: ".format(num_rnd_batch), adversary_loss.item())
            print("Classifier {} Mini-Batch loss: ".format(num_rnd_batch), classifier_loss.item())
            print("Total {} Mini-Batch loss: ".format(num_rnd_batch), total_classifier_loss.item())

      except:
        flag = False
        self.de_freez_model_weights(self.adversary)

      # Unfreeze weights of the adversary model
      self.de_freez_model_weights(self.adversary)

    # set the path ----------------------------------------------------------
    def save_best_model(self,path,metrics):

      torch.save(metrics, path +self.model_name)



    def train_eval(self, train_loader ,validation_loader,train_set,fairness_metric = "equ_odds_percent",max_shift_acc = 0.07,ippts_loader = None,fine_terms_list = None, iterations = 15,epochs = 1, num_mini_batch =1,noise_size =100, selection_score = "eq_odds",path ="", model_name = "",transformer_name = ""):




        best_fairness_metric = 0

        print("\n")
        print('Validation metrics bfore debiasing:')

        performance_log = self.conduct_validation(validation_loader,-1,save = False)

        max_best_accuracy = performance_log['Accuracy']



        print("*"*10,"Start Debiasing Stage","*"*10, end = "\n")

        for iteration in range(iterations):  # loop over the dataset multiple times
          print("*"*10,"Epoch","*"*10,iteration+1 )

          start_time = time.time()
          self.train_classifier_rnd_batch(train_set, num_mini_batch,noise_size = 100)


          self.train_adversary(train_loader, epochs)
          end_time = time.time()

          train_iteration_duration = start_time - end_time
          # print("train_iteration_duration",train_iteration_duration)


          # print('Validation metrics:')
          # try:
          #
          #   performance_log = self.conduct_validation(validation_loader,iteration)
          #
          # except:
          #   print("Model Diverged")



          if ippts_loader != None:

                print("Fine_term Results")
                performance_log,performance_log_fine_terms  = self.conduct_validation_fine_term(ippts_loader,iteration,fine_terms_list)


                print("\n")

          scaled_max_shift_accuracy = self.selection_score_computer(max_best_accuracy ,max_shift_acc)

          if abs(max_best_accuracy - performance_log['Accuracy']) <= scaled_max_shift_accuracy and  (performance_log["gender_"+fairness_metric] + performance_log["race_"+fairness_metric]) > best_fairness_metric:


              print("BEST previous fair score:",best_fairness_metric ,"max_best_accuracy:"
              ,max_best_accuracy,"New Accuracy",performance_log["Accuracy"],"Epoch",iteration+1)

              best_fairness_metric = performance_log["gender_"+fairness_metric] + performance_log["race_"+fairness_metric]


              best_model_metrics ={}
              # best_model_metrics["classifier"] = self.classifier.state_dict()
              # best_model_metrics["adversary"] = self.adversary.state_dict()
              best_model_metrics["EPOCH"] = iteration+1
              self.best_epoch = iteration+1
              self.save_best_model(path, best_model_metrics)

              best_perofrmance_log = {}
              best_perofrmance_log_fair = {}

              best_perofrmance_log = {}
              best_perofrmance_log_fine = {}

              best_perofrmance_log = performance_log
              if ippts_loader != None:
                best_perofrmance_log_fine = performance_log_fine_terms

       #----self.path is for NDABERT but we want path to fairNDABERT is given as path-------------------------------------------------------------------------------------------
        self.log_training.save_log(path)
        if ippts_loader != None:
          self.log_training.save_log(path,True)

        self.log_training.save_best_log(path,best_perofrmance_log,fine_term= False)
        if ippts_loader != None:
          self.log_training.save_best_log(path,best_perofrmance_log_fine,fine_term= True)

        epoch_coarse = self.log_training.get_epoch_coarse()
        if ippts_loader != None:
          epoch_fine = self.log_training.get_epoch_fine()


        self.plot_training.plot_iteration_score(epoch_coarse,["Accuracy","race_equ_odds_percent","gender_equ_odds_percent","race_p_value","gender_p_value"],best_model_metrics["EPOCH"])
        self.plot_training.plot_iteration_score(epoch_coarse,["race_equ_odds","gender_equ_odds"],best_model_metrics["EPOCH"])

        if ippts_loader != None:

          self.plot_training.plot_iteration_score_fine_grain(epoch_fine, "equ_odds_percent", best_model_metrics["EPOCH"],"gender")
          self.plot_training.plot_iteration_score_fine_grain(epoch_fine, "equ_odds_percent", best_model_metrics["EPOCH"],"race")



        return best_perofrmance_log, best_perofrmance_log_fine,epoch_coarse,epoch_fine



    def freez_model_weights(self,model):
      for param in model.parameters():
        param.requires_grad = False

    def de_freez_model_weights(self, model):
      for param in model.parameters():
        param.requires_grad = True

    def set_seed_val(self, seed_val):
      #------------------------------------------------------------------------------
      random.seed(seed_val)
      np.random.seed(seed_val)
      torch.manual_seed(seed_val)
      if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
      #------------------------------------------------------------------------------

    # set the path ----------------------------------------------------------
    def save_best_model(self,path,metrics):

      torch.save(metrics, path +self.model_name)
