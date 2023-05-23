            
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
#!pip install sentencepiece
import pandas as pd

import sys
sys.path.append('/content/drive/MyDrive/Master_thesis/Metrics_Evaluations')
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
#________________________________________________________________________________
##Set random values
# seed_val = 42
# random.seed(seed_val)
# np.random.seed(seed_val)
# torch.manual_seed(seed_val)
# if torch.cuda.is_available():
#   torch.cuda.manual_seed_all(seed_val)
#________________________________________________________________________________

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


import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.ERROR)

import time

# address
import sys
sys.path.append('/content/drive/MyDrive/SS_Fair/Models')
from  Arch import noise_gen


import sys
sys.path.append('/content/drive/MyDrive/SS_Fair/Models')
from  Parent_Models import TextClassificationModel, DebiasingMethod, TrainingLog, PlottingUtils


import os

# GANBERT class that inherits from TextClassificationModel
class GANBERT(TextClassificationModel):

    def __init__(self, transformer, discriminator, generator, adversary, supervised_loss, loss_adversary, log_training,plot_training,seed,discriminator_LR=2e-5, generator_LR=1e-2, adversary_LR=0.001, lda=[15, 30], path="", model_name="", transformer_name=""):

        super().__init__(log_training,plot_training)

        self.transformer = transformer
        self.discriminator = discriminator
        self.generator = generator
        self.adversary = adversary
        self.seed= seed
        self.set_seed_val(self.seed)

        ####################################################################

        transformer_vars = [i for i in self.transformer.parameters()]
        d_vars = transformer_vars + [v for v in self.discriminator.parameters()]
        g_vars = [v for v in self.generator.parameters()]


        self.dis_optimizer = torch.optim.AdamW(d_vars, lr=discriminator_LR)
        self.gen_optimizer = torch.optim.AdamW(g_vars, lr=generator_LR)
        self.optimizer_adversary =torch.optim.Adam(self.adversary.parameters(), lr= adversary_LR)

        #######################################################################

        # self.discriminator_LR = discriminator_LR
        # self.generator_LR = generator_LR
        # self.adversary_LR = adversary_LR
        self.supervised_loss = supervised_loss
        self.loss_adversary = loss_adversary
      
        self.lda = lda
        self.path = path
        self.model_name = model_name
        self.transformer_name = transformer_name

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

    
    def get_adv(self):

      return self.adversary

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

      self.transformer.eval() 
      self.discriminator.eval()
      self.generator.eval()
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
              model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
              hidden_states = model_outputs.last_hidden_state[:,0]
              _, logits, probs = self.discriminator(hidden_states)
              filtered_logits = logits[:,0:-1]

          _, preds = torch.max(filtered_logits, 1)
          '''
          0,1 coverter
          '''
          # This will map all the 2s to 1s and all other values to 0s.
          label_ids = (b_labels.to('cpu').numpy() == 2).astype(int)
          pred = (preds.detach().to('cpu').numpy() == 2).astype(int)
          gender_label = gender_label.cpu().numpy()
          race_label = race_label.cpu().numpy()
          y_score = probs[:,1:2]
          # print("y_score",y_score.shape)
          # print("truths",label_ids.shape)

          data_dict["predictions_net"].extend(pred)
          data_dict["truths"].extend(label_ids)
          data_dict["identities_gender"].extend(gender_label)
          data_dict["identities_race"].extend(race_label)
          data_dict["y_scores"].extend(y_score.cpu().numpy())
      
      return self.performance_metrics(data_dict, epoch,mode)


    def conduct_validation_fine_term(self,data_loader,epoch, fine_terms_list,mode = "Testing"):
  
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
    

      self.discriminator.eval()
      self.transformer.eval()
      self.generator.eval()
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
              model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
              hidden_states = model_outputs.last_hidden_state[:,0]
              _, logits, probs = self.discriminator(hidden_states)
              filtered_logits = logits[:,0:-1]

          _, preds = torch.max(filtered_logits, 1)

          label_ids = (b_labels.to('cpu').numpy() == 2).astype(int)
          pred = (preds.detach().to('cpu').numpy() == 2).astype(int)
    
          gender_label = gender_label.cpu().numpy()
          race_label = race_label.cpu().numpy()
          y_score = probs[:,1:2]
          # print("y_score",y_score.shape)

        

          data_dict["predictions_net"].extend(pred)
          data_dict["truths"].extend(label_ids)
          data_dict["identities_gender"].extend(gender_label)
          data_dict["identities_race"].extend(race_label)
          data_dict["y_scores"].extend(y_score.cpu().numpy())
          data_dict["fine_terms"].extend(fine_term)

      # self.performance_metrics_fine_term(data_dict)
      return self.performance_metrics(data_dict, epoch,mode),self.performance_metrics_fine_term(data_dict, epoch,fine_terms_list,mode)     


    def pretrain_classifier(self,train_loader,l = 0.85,noise_size = 100):

      """
      Pretrains the classifier by training the generator and discriminator.

      Parameters:
          train_loader (DataLoader): DataLoader object for training data.
          l (float): hyperparameter for controlling the interpolation rate between 
              generator's output and transformer's output.
          noise_size (int): size of the noise input for the generator.

      Returns:
          None
      """

      self.freez_model_weights(self.adversary)

      self.transformer.train()
      self.generator.train()
      self.discriminator.train()

      # Reset the total loss for this epoch.
      tr_g_loss = 0
      tr_d_loss = 0

      for step, batch in enumerate(train_loader):


          b_input_ids = batch['input_ids'].to(device)
          b_input_mask = batch['attention_mask'].to(device)
          b_labels = batch['targets'].to(device)
          b_label_mask = batch['label_mask'].to(device)
          gender_label = batch['identity_gender_term'].to(device)
          race_label = batch['identity_race_term'].to(device)

          #---------------------------------
          # Real data Transformer + Discriminator
          #---------------------------------

          model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
          hidden_states = model_outputs.last_hidden_state[:,0]
          NDA = hidden_states
          real_feature, real_logits, real_probs = self.discriminator(hidden_states)

          #---------------------------------
          # Simple Generator + NDA+ Discriminator
          #---------------------------------

          #fake data the same batch size of unlabel data
          noise = Arch.noise_gen(b_input_ids.shape[0], noise_size, device)
          # Gnerate Fake data
          gen_rep = self.generator(noise)

          # if l == -1:
          #   alpha = 0.9
          #   l = np.random.beta(alpha, alpha)
          #   l = max(l, 1-l)


          # neg_aug = l * gen_rep + (1 - l) * NDA
          # neg_aug = neg_aug.to(device)

          fake_feature, fake_logits, fake_probs  = self.discriminator(gen_rep)

          #---------------------------------
          # Generator's LOSS estimation
          #---------------------------------
          g_loss_d =  -1 * torch.mean(torch.log(1 - fake_probs[:, -1] + epsilon))

          g_feat_reg = torch.mean(torch.pow(torch.mean(real_feature, dim=0) - torch.mean(fake_feature, dim=0), 2))
          g_loss = g_loss_d + g_feat_reg




          #---------------------------------
          # Discriminator's LOSS estimation
          #---------------------------------
          D_L_Supervised = self.supervised_loss(real_logits,b_labels,b_label_mask)
          D_L_unsupervised1U = -1 * torch.mean(torch.log(1 - real_probs[:, -1] + epsilon))
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
          d_loss.backward()

          # Apply modifications
          self.gen_optimizer.step()
          self.dis_optimizer.step()

          # Save the losses to print them later
          tr_g_loss += g_loss.item()
          tr_d_loss += d_loss.item()


      # Calculate the average loss over all of the batches.
      ###### it is very important how many times we use label data and how we calculate the loss
      avg_train_loss_g = tr_g_loss / len(train_loader)
      avg_train_loss_d = tr_d_loss/ len(train_loader)

      # Measure how long this epoch took.

      print("")
      print("  Average training loss generetor: {0:.3f}".format(avg_train_loss_g))
      print("  Average training loss discriminator: {0:.3f}".format(avg_train_loss_d))

      result_dic = {

          'Training Loss generator': avg_train_loss_g,
          'Training Loss discriminator sup': avg_train_loss_d,
      }

      self.de_freez_model_weights(self.adversary)


      return result_dic
    
    def pretrain_adversary(self,train_loader,epochs):

        """
        Pretrains the adversary model using the training data in train_loader.
        The model is trained for a number of epochs defined by the epochs parameter.
        The performance of the model is tracked by the pretrain_adversary_loss variable.
        """

        self.freez_model_weights(self.transformer)
        self.freez_model_weights(self.discriminator)
        self.freez_model_weights(self.generator)

        self.transformer.eval()
        self.discriminator.eval()
        self.generator.eval()

        self.adversary.train()

        adv_pred_gender= np.empty((0,))
        adv_pred_race= np.empty((0,))

        adv_truth_gender= np.empty((0,))
        adv_truth_race= np.empty((0,))

        data_dict = {"adv_pred_gender": [], "adv_pred_race": [], "adv_truth_gender": [], "adv_truth_race": []}


        print("Adversary Model")
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

              model_outputs = self.transformer(input_ids,attention_mask)
              hidden_states = model_outputs.last_hidden_state[:,0]

              # This is the place that makes difference between different version of the adversarial debiasing model
              ######################################################################
              classifier_prev_output, real_logits, real_probs = self.discriminator(hidden_states)
              adversary_output_gender,adversary_output_race = self.adversary(classifier_prev_output)
              ######################################################################


              adversary_loss_gender = self.loss_adversary(adversary_output_gender, identity_term_gender) # compute loss
              adversary_loss_race = self.loss_adversary(adversary_output_race, identity_term_race) # compute loss
              adversary_loss = self.lda[0]*adversary_loss_gender + self.lda[1]*adversary_loss_race

              self.optimizer_adversary.zero_grad()
              adversary_loss.backward() # back prop
              self.optimizer_adversary.step()

              pretrain_adversary_loss += adversary_loss.item()
              epoch_loss += adversary_loss.item()
              epoch_batches += 1
              steps += 1

              _, gender_preds = torch.max(adversary_output_gender, 1)
              _, race_preds = torch.max(adversary_output_race, 1)



              gender_preds = gender_preds.detach().cpu().numpy()
              race_preds = race_preds.to('cpu').numpy()

              data_dict["adv_pred_gender"].extend(identity_term_gender.cpu().numpy())
              data_dict["adv_pred_race"].extend(identity_term_race.cpu().numpy())
              data_dict["adv_truth_gender"].extend(gender_preds)
              data_dict["adv_truth_race"].extend(race_preds)



          print("Average Pretrain Adversary epoch loss: ", epoch_loss/epoch_batches)
        print("Average Pretrain Adversary batch loss: ", pretrain_adversary_loss/steps)

        self.de_freez_model_weights(self.transformer)
        self.de_freez_model_weights(self.discriminator)
        self.de_freez_model_weights(self.generator)

        gender_adv_metrics =  metrics_eval.Performance_Metrics.metric_cal(data_dict["adv_truth_gender"],  data_dict["adv_pred_gender"])
        race_adv_metrics = metrics_eval.Performance_Metrics.metric_cal( data_dict["adv_truth_race"], data_dict["adv_pred_race"])

        print("gender acc test: ", gender_adv_metrics)
        print("race acc test: ",race_adv_metrics)


        print("----------------------------------------------------------------------------")

 
        # self.race_adv_metrics.append(race_adv_metrics[1])
        # self.gender_adv_metrics.append(gender_adv_metrics[1])

        return  self.adversary


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
          self.pretrain_classifier(train_loader,l,noise_size)

          print('Validation metrics:')
          # performance_log = self.conduct_validation(validation_loader,iteration)

          print("Train Adversay", end = '\n')
          self.adversary = self.pretrain_adversary(train_loader,epochs_adv)
          
          print("\n")

          if ippts_loader != None and iteration > -1:

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
                  best_model_metrics["discriminator"] = self.discriminator.state_dict()
                  best_model_metrics["adversary"] = self.adversary.state_dict()
                  best_model_metrics["generator"] = self.generator.state_dict()


                  print("performance_log",performance_log)

                  best_model_metrics["EPOCH"] = iteration+1


                  self.save_best_model(best_model_metrics)
                  self.transformer.save_pretrained(self.path+self.transformer_name)
                  best_perofrmance_log = {}
                  best_perofrmance_log_fine = {}

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



    
    # set the path ----------------------------------------------------------
    def save_best_model(self,metrics):

      torch.save(metrics, self.path +self.model_name)


class FairGANBERT(DebiasingMethod):

    def __init__(self,adversary, supervised_loss, loss_adversary, log_training,plot_training,seed,discriminator_LR=2e-5, generator_LR=1e-2, adversary_LR=0.001,batch_size = 8, lda=[15, 30],flag_transformer = False, path="", model_name="", transformer_name=""):


        super().__init__(log_training,plot_training)


        #address
        checkpoint = torch.load(path+model_name)

        transformer = DistilBertModel.from_pretrained(path+transformer_name)
        config = AutoConfig.from_pretrained('distilbert-base-uncased')
        hidden_size = int(config.hidden_size)

        discriminator = Arch.Discriminator(input_size=hidden_size, hidden_sizes=[768], num_labels=3 , dropout_rate=0.3)
        discriminator.load_state_dict(checkpoint['discriminator'])

        adversary.load_state_dict(checkpoint['adversary'])

        adversary.to(device)
        discriminator.to(device)
        transformer.to(device)

        print("BEST epoch is: ",checkpoint['EPOCH'])
        generator = Arch.Generator(noise_size=100, output_size=hidden_size, hidden_sizes=[768], dropout_rate=0.3)
        generator.to(device)


        ##############

        self.transformer = transformer
        self.discriminator = discriminator
        self.generator = generator
        self.adversary = adversary

        # ********
        #either apply debiasing to BERT or not
        if flag_transformer == True:
          transformer_vars = [i for i in self.transformer.parameters()]
        else:
          transformer_vars = []

        d_vars = transformer_vars + [v for v in self.discriminator.parameters()]
        g_vars = [v for v in self.generator.parameters()]

        dis_optimizer = torch.optim.AdamW(d_vars, lr=discriminator_LR)
        gen_optimizer = torch.optim.AdamW(g_vars, lr=generator_LR)
        optimizer_adv =  torch.optim.Adam(self.adversary.parameters(), lr= adversary_LR)

        self.optimizer_discriminator = dis_optimizer
        self.optimizer_generator = gen_optimizer
        self.optimizer_adversary = optimizer_adv

        self.supervised_loss = supervised_loss
        self.loss_adversary = loss_adversary

        

        self.path = path
        self.lda = lda
        self.batch_size = batch_size
        # self.lbda = lbda
        self.model_name = model_name
        self.transformer_name = transformer_name

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

      self.transformer.eval() 
      self.discriminator.eval()
      self.generator.eval()
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
              model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
              hidden_states = model_outputs.last_hidden_state[:,0]
              _, logits, probs = self.discriminator(hidden_states)
              filtered_logits = logits[:,0:-1]

          _, preds = torch.max(filtered_logits, 1)
          '''
          0,1 coverter
          '''
          # This will map all the 2s to 1s and all other values to 0s.
          label_ids = (b_labels.to('cpu').numpy() == 2).astype(int)
          pred = (preds.detach().to('cpu').numpy() == 2).astype(int)
          gender_label = gender_label.cpu().numpy()
          race_label = race_label.cpu().numpy()
          y_score = probs[:,1:2]
    
          data_dict["predictions_net"].extend(pred)
          data_dict["truths"].extend(label_ids)
          data_dict["identities_gender"].extend(gender_label)
          data_dict["identities_race"].extend(race_label)
          data_dict["y_scores"].extend(y_score.cpu().numpy())
      
      return self.performance_metrics(data_dict, epoch,mode,save)


    def conduct_validation_fine_term(self,data_loader,epoch,fine_terms_list,save = True, mode = "Testing"):
  
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
    

      self.discriminator.eval()
      self.transformer.eval()
      self.generator.eval()
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
              model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
              hidden_states = model_outputs.last_hidden_state[:,0]
              _, logits, probs = self.discriminator(hidden_states)
              filtered_logits = logits[:,0:-1]

          _, preds = torch.max(filtered_logits, 1)

          label_ids = (b_labels.to('cpu').numpy() == 2).astype(int)
          pred = (preds.detach().to('cpu').numpy() == 2).astype(int)
    
          gender_label = gender_label.cpu().numpy()
          race_label = race_label.cpu().numpy()
          y_score = probs[:,1:2]
          # print("y_score",y_score.shape)

        

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

      self.freez_model_weights(self.transformer)
      self.freez_model_weights(self.discriminator)

      for epoch in range(epochs):

        for step, batch in enumerate(train_loader):

            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            b_label_mask = batch['label_mask'].to(device)
            identity_term_gender = batch['identity_gender_term'].to(device)
            identity_term_race = batch['identity_race_term'].to(device)


            model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
            hidden_states = model_outputs.last_hidden_state[:,0]
            classifier_prev_output, real_logits, real_probs = self.discriminator(hidden_states)

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

      self.de_freez_model_weights(self.transformer)
      self.de_freez_model_weights(self.discriminator)

    def train_classifier_rnd_batch(self,train_set, num_rnd_batch =2,noise_size = 100):

      # Freeze weights of the adversary model
      self.freez_model_weights(self.adversary)

      # Set the discriminator and transformer models to train mode
      self.discriminator.train()
      # it does not have effect if flag_transformer is False
      self.transformer.train()

      # Initialize variables
      pretrain_classifier_loss = 0
      steps = 0

      # Create data loader for loading data in random batches
      rnd_batch_gender_loader = data_loaders.rnd_batch_ss_gan_data_loader(train_set ,self.batch_size, num_rnd_batch)
      print(len(rnd_batch_gender_loader))
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

            # Pass the input data through the transformer model to get the hidden states
            model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
            hidden_states = model_outputs.last_hidden_state[:,0]

            # Pass the hidden states through the discriminator model to get real feature, logits and probabilities
            real_feature, real_logits, real_probs = self.discriminator(hidden_states)

            # Pass the real feature through the adversary model to get the output for gender and race
            adversary_output_gender,adversary_output_race = self.adversary(real_feature)

            # Compute the loss for the adversary model for gender and race
            adversary_loss_gender = self.loss_adversary(adversary_output_gender, identity_term_gender)
            adversary_loss_race = self.loss_adversary(adversary_output_race, identity_term_race)

            # Compute the total loss for the adversary model
            adversary_loss = self.lda[0]*adversary_loss_gender + self.lda[1]*adversary_loss_race

            # Compute the supervised loss for the discriminator model
            D_L_Supervised = self.supervised_loss(real_logits,b_labels,b_label_mask)

            # Compute the unsupervised loss for the discriminator model
            D_L_unsupervised1U = -1 * torch.mean(torch.log(1 - real_probs[:, -1] + epsilon))

            # Compute the total loss for the discriminator model
            d_loss = D_L_Supervised + D_L_unsupervised1U - adversary_loss

            # Perform optimization for the discriminator model
            self.optimizer_discriminator.zero_grad()
            d_loss.backward()
            self.optimizer_discriminator.step()

      except:
        flag = False
        self.de_freez_model_weights(self.adversary)
        print("No mini training batch left ", end = "\n")

      # Unfreeze weights of the adversary model
      self.de_freez_model_weights(self.adversary)

    # set the path ----------------------------------------------------------
    def save_best_model(self,path,metrics):

      torch.save(metrics, path +self.model_name)



    def train_eval(self, train_loader ,validation_loader,train_set,fairness_metric = "equ_odds_percent",max_shift_acc = 0.07,ippts_loader = None,fine_terms_list = None, iterations = 15,epochs = 1, num_mini_batch =1,noise_size =100, selection_score = "eq_odds",path ="", model_name = "",transformer_name = ""):



       
        best_fairness_metric = 0
        best_perofrmance_log = {}
        best_perofrmance_log_fine = {}
        
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
                performance_log, performance_log_fine_terms  = self.conduct_validation_fine_term(ippts_loader,iteration,fine_terms_list)
                

                print("\n")

          scaled_max_shift_accuracy = self.selection_score_computer(max_best_accuracy ,max_shift_acc)

          if abs(max_best_accuracy - performance_log['Accuracy']) <= scaled_max_shift_accuracy and  (performance_log["gender_"+fairness_metric] + performance_log["race_"+fairness_metric]) > best_fairness_metric:
          
          
              print("BEST previous fair score:",best_fairness_metric ,"max_best_accuracy:"
              ,max_best_accuracy,"New Accuracy",performance_log["Accuracy"],"Epoch",iteration+1)

              best_fairness_metric = performance_log["gender_"+fairness_metric] + performance_log["race_"+fairness_metric]

              
              best_model_metrics ={}
              # best_model_metrics["discriminator"] = self.discriminator.state_dict()
              # best_model_metrics["adversary"] = self.adversary.state_dict()
              # best_model_metrics["generator"] = self.generator.state_dict()
              best_model_metrics["EPOCH"] = iteration+1
              self.best_epoch = iteration+1
              self.save_best_model(path, best_model_metrics)
              self.transformer.save_pretrained(path+self.transformer_name)

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

      
