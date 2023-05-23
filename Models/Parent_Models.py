
'''
class TextClassificationModel
class DebiasingMethod
class TrainingLog
class PlottingUtils
class Plot_data
'''


# !pip install transformers==4.3.2
# !pip install transformers==4.26.0



import torch
import io
import random
import time
import math
import datetime
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.ERROR)
import ast
import os

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from collections import defaultdict
from matplotlib.patches import Rectangle

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import matthews_corrcoef
# import sentencepiece

import torch.nn.functional as F
import torch.nn as nn
from transformers import *

#!pip install sentencepiece
from sklearn import metrics



import sys
sys.path.append('/project/6001054/shah92/SS-Fair/Metrics_Evaluations/')
import metrics_eval

# address
import sys
sys.path.append('/project/6001054/shah92/SS-Fair/Models')
import Arch

import sys
sys.path.append('/project/6001054/shah92/SS-Fair/Dataloaders/')
import data_loaders
import Data_sampler

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


import shutil



from prettytable import PrettyTable
import csv

import itertools

from matplotlib.pylab import True_

from pandas.core.describe import describe_timestamp_as_categorical_1d
import seaborn as sns


import seaborn as sns

class TextClassificationModel:

    def __init__(self,log_training,plot_training):
        '''
        it comes from the FairTextClassification
        '''

        self.log_training = log_training
        self.plot_training = plot_training


    def performance_metrics(self,data_dict,epoch,mode):

      performance_metrics = metrics_eval.Performance_Metrics.get_metrics(data_dict["truths"], data_dict["predictions_net"],data_dict['identities_gender'],data_dict['identities_race'])
      try: 
        fpr, tpr, _ = metrics.roc_curve(data_dict["truths"], data_dict["y_scores"], pos_label=1)
      except:
        print(len(data_dict["truths"]), len(data_dict["y_scores"]))

      roc_auc = metrics.auc(fpr, tpr)
      confusion_matrix = metrics.confusion_matrix(data_dict["truths"], data_dict["predictions_net"])
      metrics_dic_val_gender = metrics_eval.Metrics(data_dict["predictions_net"], data_dict["truths"], data_dict["identities_gender"], thres).all_metrics("gender")
      metrics_dic_val_race = metrics_eval.Metrics(data_dict["predictions_net"], data_dict["truths"], data_dict["identities_race"], thres).all_metrics("race")

      performance_log = {**performance_metrics, **metrics_dic_val_gender, **metrics_dic_val_race}
      performance_log["roc_auc"] = roc_auc
      performance_log["confusion_matrix"] = confusion_matrix
      performance_log["Mode"] = mode


      self.print_performance_log(performance_log)

      # object of TrainingLog
      self.log_training.update_epoch_log( performance_log, epoch)
      '''
      You must defined a object of PrettyTable as you initialized your model in FairTextClassification
      '''
      return performance_log

    def performance_metrics_fine_term(self, data_dict, epoch,fine_terms_list ,mode):

      predictions_net = data_dict['predictions_net']
      truths = data_dict['truths']
      identities_gender = data_dict['identities_gender']
      identities_race = data_dict['identities_race']
      fine_terms = data_dict['fine_terms']

      print("finter_function",fine_terms_list)
      # fine_terms_list = ['gender', 'race']
      
      # race_term = set(["African", "Arab", "Asian", "Caucasian", "Hispanic","Refugee"])
      # religion_term =  set(["Islam", "Buddhism", "Jewish","Hindu", "Christian"])
      # gender_term =  set(["Men", "Women","Homosexual"])

      # fine_terms_list = [gender_term,race_term]
      thres = 0.5

      # Initialize dictionary to store performance metrics for each fine-grained term
      fine_term_metrics = {'gender': {}, 'race': {}}

      # Compute performance metrics for each fine-grained term
      fine_gender_metrics = metrics_eval.Metrics_fine_terms(predictions_net, truths, identities_gender, fine_terms, fine_terms_list[0], data_dict['y_scores'],thres).all_metrics_terms()
      fine_race_metrics = metrics_eval.Metrics_fine_terms(predictions_net, truths, identities_race, fine_terms, fine_terms_list[1], data_dict['y_scores'],thres).all_metrics_terms()

      # Add performance metrics to dictionary
      fine_term_metrics['gender'] = fine_gender_metrics
      fine_term_metrics['race'] = fine_race_metrics
      fine_term_metrics["Mode"] = mode

      self.print_performance_log(fine_term_metrics)

      # object of TrainingLog
      self.log_training.update_epoch_log( fine_term_metrics, epoch, True)

      return fine_term_metrics

      
    def save_log_epoch(self):

       self.log_training

    '''
    must go to print class
    '''
    def print_performance_log(self,performance_log):

        table = PrettyTable()
        table.field_names = []

        # Add the performance metrics
        for key, value in performance_log.items():
            table.add_row([key, value])

        print(table)




class DebiasingMethod:

    def __init__(self,log_training,plot_training):
        '''
        it comes from the FairTextClassification
        '''

        self.log_training = log_training
        self.plot_training = plot_training


    def selection_score_computer(self,best_accuracy ,max_shift_acc = 0.07):

      scaled_max_shift_acc = max_shift_acc*best_accuracy

      return scaled_max_shift_acc




    def performance_metrics(self,data_dict,epoch,mode, save = True):

      performance_metrics = metrics_eval.Performance_Metrics.get_metrics(data_dict["truths"], data_dict["predictions_net"],data_dict['identities_gender'],data_dict['identities_race'])
      try: 
        fpr, tpr, _ = metrics.roc_curve(data_dict["truths"], data_dict["y_scores"], pos_label=1)
      except:
        print(len(data_dict["truths"]), len(data_dict["y_scores"]))

      roc_auc = metrics.auc(fpr, tpr)
      confusion_matrix = metrics.confusion_matrix(data_dict["truths"], data_dict["predictions_net"])
      metrics_dic_val_gender = metrics_eval.Metrics(data_dict["predictions_net"], data_dict["truths"], data_dict["identities_gender"], thres).all_metrics("gender")
      metrics_dic_val_race = metrics_eval.Metrics(data_dict["predictions_net"], data_dict["truths"], data_dict["identities_race"], thres).all_metrics("race")

      performance_log = {**performance_metrics, **metrics_dic_val_gender, **metrics_dic_val_race}
      performance_log["roc_auc"] = roc_auc
      performance_log["confusion_matrix"] = confusion_matrix
      performance_log["Mode"] = mode


      self.print_performance_log(performance_log)

      # object of TrainingLog
      if save == True:
        self.log_training.update_epoch_log( performance_log, epoch)
     
      return performance_log
    
    def performance_metrics_fine_term(self, data_dict, epoch,fine_terms_list ,mode):

      predictions_net = data_dict['predictions_net']
      truths = data_dict['truths']
      identities_gender = data_dict['identities_gender']
      identities_race = data_dict['identities_race']
      fine_terms = data_dict['fine_terms']

      print("finter_function",fine_terms_list)
      # fine_terms_list = ['gender', 'race']
      
      # race_term = set(["African", "Arab", "Asian", "Caucasian", "Hispanic","Refugee"])
      # religion_term =  set(["Islam", "Buddhism", "Jewish","Hindu", "Christian"])
      # gender_term =  set(["Men", "Women","Homosexual"])

      # fine_terms_list = [gender_term,race_term]
      thres = 0.5

      # Initialize dictionary to store performance metrics for each fine-grained term
      fine_term_metrics = {'gender': {}, 'race': {}}

      # Compute performance metrics for each fine-grained term
      fine_gender_metrics = metrics_eval.Metrics_fine_terms(predictions_net, truths, identities_gender, fine_terms, fine_terms_list[0], data_dict['y_scores'],thres).all_metrics_terms()
      fine_race_metrics = metrics_eval.Metrics_fine_terms(predictions_net, truths, identities_race, fine_terms, fine_terms_list[1], data_dict['y_scores'],thres).all_metrics_terms()

      # Add performance metrics to dictionary
      fine_term_metrics['gender'] = fine_gender_metrics
      fine_term_metrics['race'] = fine_race_metrics
      fine_term_metrics["Mode"] = mode

      self.print_performance_log(fine_term_metrics)

      # object of TrainingLog
      self.log_training.update_epoch_log( fine_term_metrics, epoch, True)

      return fine_term_metrics

      

      
    # def save_log_epoch(self):

    #    self.log_training

    # '''
    # must go to print class
    # '''
    def print_performance_log(self,performance_log):

        table = PrettyTable()
        table.field_names = []

        # Add the performance metrics
        for key, value in performance_log.items():
            table.add_row([key, value])

        # table.set_row_style(1, "background-color: yellow")

        print(table)



    def remove_model(self,path):
      if os.path.isfile(path):
          os.remove(path)
      elif os.path.isdir(path):
          shutil.rmtree(path)
      else:
          raise ValueError(f"Invalid path {path}")





class TrainingLog:
  
    def __init__(self, model_name, seed, num_label_data, num_unlabel_data):
        self.model_name = model_name
        self.seed = seed
        self.num_label_data = num_label_data
        self.num_unlabel_data = num_unlabel_data
        '''
        you probably need to change it in a way that it is a dic (seed) of dic(epoch) of dic(metrics)
        '''
        self.epoch_logs = {}
        self.epoch_logs_fine_terms = {}

        '''
        add each seed best model dictionary, a dic (seed) of a dic (best model)
        '''
        self.best_model = {}

        self.epoch_coarse = {}
        self.epoch_fine = {}


    '''
    all must change to include seed, also add a function to update the best model
    '''    
        
    def update_epoch_log(self, data, epoch, fine_term = False):

       if fine_term == True:
        
          self.epoch_fine[epoch] = data


          table = self.epoch_logs_fine_terms.get(epoch, None)
          if table is None:
              table = PrettyTable()
              table.field_names = []
              self.epoch_logs_fine_terms[epoch] = table
          
          # Add the columns using the keys of the dictionary
          if not table.field_names:
              table.field_names = list(data.keys())
          
          # Add the values as a row
          table.add_row(list(data.values()))

       else:
          self.epoch_coarse[epoch] = data
          table = self.epoch_logs.get(epoch, None)
          if table is None:
              table = PrettyTable()
              table.field_names = []
              self.epoch_logs[epoch] = table
           
          # Add the columns using the keys of the dictionary
          if not table.field_names:
              table.field_names = list(data.keys())
          
          # Add the values as a row
          table.add_row(list(data.values()))
          

    def get_epoch_coarse(self):
      return self.epoch_coarse


    def get_epoch_fine(self):
      return self.epoch_fine

        




    def save_log(self, path,fine_term= False):

      csv_file = f"{self.model_name}_seed{self.seed}_label{self.num_label_data}_unlabel{self.num_unlabel_data}.csv"


      if fine_term == False:
      
        epoch = list( self.epoch_coarse.keys())
        # # print("epoch",epoch)

        # extract the column names (keys of the inner dictionaries)
        columns = list( self.epoch_coarse[epoch[0]].keys())
        # print("columns",columns)

        data = {col: [self.epoch_coarse[epoch[i]][col] for i in range(len(epoch))] for col in columns}
        print("data",data)

        df = pd.DataFrame(data, columns=columns)

        # Add the 'Epoch' column
        df.insert(0, 'Epoch', range(1, len(epoch)+1))

        # Write the dataframe to a CSV file
        df.to_csv(path+csv_file, index=False)




      else:



        # Create an empty dataframe
        df = pd.DataFrame()

        # Get the keys of the outer dictionary (0 or 1)
        epoch = list(self.epoch_fine.keys())

        # Extract the column names (keys of the inner dictionaries) for the 'gender' key
        columns_gender = self.epoch_fine[epoch[0]]['gender']
        prefix_columns_gen = [col for col in columns_gender.keys()]
        suffix_col_gen = [col for col in list(columns_gender.values())[0].keys()]
        gender_cols = [x+"_"+y for x, y in itertools.product(prefix_columns_gen, suffix_col_gen)]

        # Extract the column names (keys of the inner dictionaries) for the 'race' key
        columns_race = self.epoch_fine[epoch[0]]['race']
        prefix_columns_race = [col for col in columns_race.keys()]
        suffix_col_race = [col for col in list(columns_race.values())[0].keys()]
        race_cols = [x+"_"+y for x, y in itertools.product(prefix_columns_race, suffix_col_race)]

        # Combine the column names for 'gender' and 'race'
        all_cols = gender_cols + race_cols

        # Add a 'Mode' column
        all_cols.append('Mode')

        # Add the key values (0 or 1) as the first column in the dataframe
        df['epoch'] = epoch
        cc = 0
        dd= 0
        # Iterate over the outer dictionary and add the values for each key to the dataframe
        for e in epoch:
            for g in prefix_columns_gen:
                for c in suffix_col_gen:
                  
                    try:
                      df["gender_"+g+"_"+c] = self.epoch_fine[e]['gender'][g][c]
                    #   dd +=1 
                    except:
                    #   cc +=1
                      pass
                      


            for r in prefix_columns_race:
                for c in suffix_col_race:
                    try:
                      df["race_"+r+"_"+c] = self.epoch_fine[e]['race'][r][c]

                    except:
                      pass
                    

            df['Mode'] = self.epoch_fine[e]['Mode']

        # Save the dataframe to a csv file
        df.to_csv(path+"fine_"+csv_file, index=False)


    def save_best_log(self, path,best_dic,fine_term= False):

      csv_file = f"BEST_{self.model_name}_seed{self.seed}_label{self.num_label_data}_unlabel{self.num_unlabel_data}.csv"


      if fine_term == False:
      
        # epoch = list( self.epoch_coarse.keys())
        # # print("epoch",epoch)

        # extract the column names (keys of the inner dictionaries)
        columns = list( best_dic.keys())
        # print("columns",columns)

        data = {col: [best_dic[col]] for col in columns}
        print("best",data)

        df = pd.DataFrame(data, columns=columns)

        print("best df",df)

        # # Add the 'Epoch' column
        # df.insert(0, 'Epoch', range(1, len(epoch)+1))

        # Write the dataframe to a CSV file
        print("path",path+csv_file)
        df.to_csv(path+csv_file, index=False)




      else:

        print("best_fine",best_dic)


        # Create an empty dataframe
        df = pd.DataFrame()

        # Get the keys of the outer dictionary (0 or 1)
        # epoch = list(self.epoch_fine.keys())

        # Extract the column names (keys of the inner dictionaries) for the 'gender' key
        columns_gender = best_dic['gender']
        # print("columns_gender",columns_gender)
        prefix_columns_gen = [col for col in columns_gender.keys()]
        # print("prefix_columns_gen",prefix_columns_gen)
        suffix_col_gen = [col for col in list(columns_gender.values())[0].keys()]
        # print("suffix_col_gen",suffix_col_gen)

        gender_cols = [x+"_"+y for x, y in itertools.product(prefix_columns_gen, suffix_col_gen)]

        # Extract the column names (keys of the inner dictionaries) for the 'race' key
        columns_race = best_dic['race']
        prefix_columns_race = [col for col in columns_race.keys()]
        suffix_col_race = [col for col in list(columns_race.values())[0].keys()]
        race_cols = [x+"_"+y for x, y in itertools.product(prefix_columns_race, suffix_col_race)]

        # Combine the column names for 'gender' and 'race'
        all_cols = gender_cols + race_cols

        # Add a 'Mode' column
        all_cols.append('Mode')

        # Add the key values (0 or 1) as the first column in the dataframe
        # df['epoch'] = epoch
        cc = 0
        dd= 0
        # Iterate over the outer dictionary and add the values for each key to the dataframe
        # for e in epoch:
        for g in prefix_columns_gen:
                for c in suffix_col_gen:
                  
                    try:
                      # print(" best_dic['gender'][g][c]", best_dic['gender'][g][c])
                      df["gender_"+g+"_"+c] = [best_dic['gender'][g][c]]
                    #   dd +=1 
                    except:
                    #   cc +=1
                      print("error", g,c)
                      pass
                      
        print("0 df",df)


        for r in prefix_columns_race:
                for c in suffix_col_race:
                    try:
                      df["race_"+r+"_"+c] = [best_dic['race'][r][c]]

                    except:
                      pass
                    

        df['Mode'] = best_dic['Mode']

        # print("best df",df)


        # Save the dataframe to a csv file
        df.to_csv(path+"fine_"+csv_file, index=False)
  

   
# PlottingUtils class contains methods for plotting the results
class PlottingUtils:

  def __init__(self,path,model_name,seed,num_label_data,num_unlabel_data):
     self.path = path
     self.model_name = model_name
     self.seed = seed
     self.num_label_data = num_label_data
     self.num_unlabel_data = num_unlabel_data


  def plot_iteration_score(self, fairness_metrics_iteration, metrics, best_epochs):

    jpg_file = f"{self.model_name}_seed{self.seed}_label{self.num_label_data}_unlabel{self.num_unlabel_data}_{metrics}.png"
    best_scores =[]
    # Create a dataframe with the metric scores and epochs
    data = {'epoch': range(1,len(fairness_metrics_iteration.keys())+1)}
    for i, metric in enumerate(metrics):
        data[metric] = [float(fairness_metrics_iteration[epoch][metric]) for epoch in fairness_metrics_iteration  for score in fairness_metrics_iteration[epoch] if score == metric ]
        best_scores.append(data[metric][best_epochs-1])

    df = pd.DataFrame(data)
    # Plot the data using Seaborn
    sns.set_style("darkgrid")
    plt.figure(figsize=(10,5))
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan', 'magenta']
    for i, metric in enumerate(metrics):
        sns.scatterplot(x=[best_epochs], y=[best_scores[i]], color='black', marker='*',s=250,zorder=3) #[best_score] plot the best_epoch
        separated = metric.split("_", 1)
        if len(separated) !=1:
          metric_formal = separated[0].capitalize() +self.title_dic(separated[1])
        else: 
          metric_formal = metric
        sns.lineplot(x='epoch', y=metric, data=df, color=colors[i], marker='o', markersize=6, label=metric_formal,zorder=1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title("Model performance per epoch")
    plt.savefig(self.path +jpg_file, bbox_inches='tight', dpi=300)
    plt.show()

  def plot_iteration_score_fine_grain(self, fairness_metrics_iterations, metric, best_epochs,acc_separation = None, type_bias= ""):

      jpg_file = f"Fine_{self.model_name}_seed{self.seed}_label{self.num_label_data}_unlabel{self.num_unlabel_data}_{metric}.png"

      color = ['black','red','orange', "blue","green","cyan","magenta","yellow","black"]
      metric_score_list = defaultdict(list)
      stats = []
      fine_terms = []
      print("fairness_metrics_iterations",fairness_metrics_iterations)
      for idx,epoch in enumerate(fairness_metrics_iterations):
          for key1 in fairness_metrics_iterations[epoch]: #iterate over first level keys (epochs)
          
              if key1 != 'Mode' and acc_separation == None:
                pass

              elif key1 != 'Mode' and acc_separation != key1:
                continue
                
                  
              elif key1 != 'Mode' and acc_separation == key1:
                pass

              else:
                continue

                
              for key2 in fairness_metrics_iterations[epoch][key1]: #iterate over second level keys (gender, race)

                            fine_term = key1+"*"+ key2 + '*' + metric
                            if fine_term not in fine_terms:

                                fine_terms.append(fine_term)
                                # print(key1,key2,metric)
                                metric_score_list[fine_term].append(fairness_metrics_iterations[epoch][key1][key2][metric])
                                stats.append(fairness_metrics_iterations[epoch][key1][key2]['number of samples'])
                            else:

                                metric_score_list[fine_term].append(fairness_metrics_iterations[epoch][key1][key2][metric])
                                stats.append(fairness_metrics_iterations[epoch][key1][key2]['number of samples'])

                      

      data = []
      print("metric_score_list",metric_score_list)
      data = pd.DataFrame(metric_score_list)
      data['epoch'] = [i+1 for i, x in enumerate(data.index)]

      for subgroup in fine_terms:
        subgroup_label = subgroup.split('*')[0].capitalize()+" " + subgroup.split('*')[1].capitalize() + self.title_dic(subgroup.split('*')[2])
        subgroup_data = data[[subgroup, 'epoch']]
        
        sns.lineplot(x='epoch', y=subgroup, data=subgroup_data, label=subgroup_label)
        sns.scatterplot(x=[best_epochs], y=[data[subgroup][best_epochs-1]], color='black', marker='*',s=250,zorder=3) #[best_score] plot the best_epoch

      label=subgroup

      print("subgroup",subgroup)
      plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

  
      plt.ylabel(self.title_dic(metric)) #set the label for y axis
      plt.xlabel('Epochs') #set the label for x-axis
      plt.title("{} Performance Per epoch{}".format(self.title_dic(metric),type_bias)) #set the title of the graph
      plt.savefig(self.path +jpg_file, bbox_inches='tight', dpi=300)
      plt.show() #display the graph

  def title_dic(self,term):


        term_dic ={"p_value":" P_value","demo_parity":" Demographic Parity","equ_odds":" Equalized Odds Difference","equ_odds_percent":" Equalized Odds Difference Percentage"
        ,"equ_opportunity":" Equalized Opportunity Difference","equ_opportunity_percent":" Equalized Opportunity Difference Percentage"}

        return term_dic.get(term,term)




import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class Plot_data:

  def __init__(self,data,path):

    self.data =data
    self.path = path
    self.data['attr'] = data['attr'].apply(ast.literal_eval)


  def path_maker(self,folder_path):

      # Check if the folder exists
      if not os.path.exists(folder_path):
        # Create a new folder
        # !mkdir $folder_path
        os.makedirs(folder_path)
        print("Folder created: ", folder_path)
      else:
        print("Folder already exists: ", folder_path)


  # def single_var_pie_chart(self,group_col = "Label", group_label_dic ={1: 'Toxic', 0: 'Non-Toxic'}, title = "add title",seed = 1000, dataset = "Noname"):

  #   # Group the data by the Label column
  #   grouped_data = self.data.groupby(group_col).size().reset_index(name='counts')
  #   # Replace the Label values with custom labels
  #   grouped_data[group_col] = grouped_data[group_col].replace(group_label_dic)

  #   # Plot the pie chart
  #   sns.set_style("darkgrid")
  #   plt.pie(grouped_data['counts'], labels=grouped_data[group_col], startangle=90, autopct=self.make_autopct(grouped_data['counts']))
  #   plt.axis('equal')
  #   # plt.legend(title=group_col, labels=[list(group_label_dic.keys())[0], list(group_label_dic.keys())[0]], loc='upper left')
  #   # -----
  #   plt.title(title)

  #   jpg_file = f"{group_col}_{dataset}_{str(seed)}.png"
  #   self.path_maker(self.path+dataset)
  #   plt.savefig(self.path+dataset+"/"+ jpg_file)

  #   plt.show()





  # def single_var_pie_chart(self, group_col="Label", group_label_dic={1: 'Toxic', 0: 'Non-Toxic'}, title="add title", seed=1000, dataset="Noname"):

  #   # Group the data by the Label column
  #   grouped_data = self.data.groupby(group_col).size().reset_index(name='counts')
  #   # Replace the Label values with custom labels
  #   grouped_data[group_col] = grouped_data[group_col].replace(group_label_dic)

  #   # Calculate the percentage for each slice
  #   total = grouped_data['counts'].sum()
  #   grouped_data['percentage'] = grouped_data['counts'] / total * 100

  #   # Plot the pie chart
  #   sns.set_style("darkgrid")
  #   patches, a,b = plt.pie(grouped_data['counts'], labels=grouped_data[group_col], startangle=90, autopct='%1.1f%%')
  #   plt.axis('equal')

  #   # Create a legend
  #   labels = []
  #   for i, row in grouped_data.iterrows():
  #       label = f"{row[group_col]}: {row['counts']} ({row['percentage']:.1f}%)"
  #       labels.append(label)
  #   plt.legend(handles=patches, labels=labels, title=group_col, loc='center right', bbox_to_anchor=(1.35, 0.5))

  #   # Add some extra styling
  #   plt.title(title, fontsize=16, fontweight='bold', y=1.05)
  #   plt.setp(patches, linewidth=0.5, edgecolor='white')
  #   colors = ['yellow', 'purple']
  #   for i in range(len(patches)):
  #       patches[i].set_facecolor(colors[i])

  #   # Save and show the plot
  #   jpg_file = f"{group_col}_{dataset}_{str(seed)}.png"
  #   self.path_maker(self.path+dataset)
  #   plt.savefig(self.path+dataset+"/"+ jpg_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
  #   plt.show()



  def single_var_pie_chart(self, group_col="Label", group_label_dic={1: 'Toxic', 0: 'Non-Toxic'}, title="add title", seed=1000, dataset="Noname"):

    # Group the data by the Label column
    grouped_data = self.data.groupby(group_col).size().reset_index(name='counts')
    # Replace the Label values with custom labels
    grouped_data[group_col] = grouped_data[group_col].replace(group_label_dic)

    # Calculate the percentage for each slice
    total = grouped_data['counts'].sum()
    grouped_data['percentage'] = grouped_data['counts'] / total * 100

    # Plot the pie chart
    sns.set_style("darkgrid")
    patches, a = plt.pie(grouped_data['counts'], startangle=90)
    plt.axis('equal')

    # Set the colors of the pie chart slices
    colors = ['orange', 'purple']
    for i in range(len(patches)):
        patches[i].set_facecolor(colors[i])

    # Create a legend
    handles = [Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(grouped_data))]
    labels = [f"{row[group_col]}: {row['counts']} ({row['percentage']:.1f}%)" for _, row in grouped_data.iterrows()]
    plt.legend(handles=handles, labels=labels, title=group_col.capitalize(), loc='center right', bbox_to_anchor=(1.35, 0.5), fontsize=11)

    # Add some extra styling
    # plt.title(title, fontsize=16, fontweight='bold', y=1.05)

    # Save and show the plot
    jpg_file = f"{group_col}_{dataset}_{str(seed)}.png"
    self.path_maker(self.path+dataset)
    plt.savefig(self.path+dataset+"/"+ jpg_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.show()




  # def single_var_pie_chart(self, group_col="Label", group_label_dic={1: 'Toxic', 0: 'Non-Toxic'}, title="Add title", seed=1000, dataset="Noname"):
  #       grouped_data = self.data.groupby(group_col).size().reset_index(name='counts')
  #       grouped_data[group_col] = grouped_data[group_col].replace(group_label_dic)

  #       def get_colors(number_of_colors):
  #           color_palette = sns.color_palette('pastel', number_of_colors)
  #           return color_palette

  #       sns.set_style("darkgrid")
  #       colors = get_colors(len(group_label_dic))
  #       patches, texts, autotexts = plt.pie(grouped_data['counts'], labels=grouped_data[group_col], colors=colors, startangle=90, autopct=self.make_autopct(grouped_data['counts']))

  #       for text, autotext in zip(texts, autotexts):
  #           text.set_color('gray')
  #           autotext.set_color('white')
  #           autotext.set_weight('bold')

  #       plt.axis('equal')
  #       plt.title(title)

  #       jpg_file = f"{group_col}_{dataset}_{str(seed)}.png"
  #       self.path_maker(self.path + dataset)
  #       plt.savefig(self.path + dataset + "/" + jpg_file)
  #       plt.show()

  # def double_var_pie_chart(self,group_col = ["Label","gender"],  group_label_dic_var1 ={1: 'Toxic', 0: 'Non-Toxic'},
  #                          group_label_dic_var2 ={1: 'Gender', 0: 'Non-Gender'}, title = "add title" ,seed = 1000, dataset = "Noname"):
    
  #   # Group the data by the Label and Gender columns
  #   grouped_data = self.data.groupby(group_col).size().reset_index(name='counts')

  #   # Replace the Label values with custom labels
  #   grouped_data[group_col[0]] = grouped_data[group_col[0]].replace(group_label_dic_var1)
  #   grouped_data[group_col[1]] = grouped_data[group_col[1]].replace(group_label_dic_var2)

  #   # Plot the pie chart
  #   sns.set_style("darkgrid")
  #   plt.pie(grouped_data['counts'], labels=grouped_data[group_col[0]] + " \\ " + grouped_data[group_col[1]].astype(str), startangle=90, autopct=self.make_autopct(grouped_data['counts']),labeldistance=3.5)
  #   plt.axis('equal')
  #   # plt.legend(title='Gender keyword\Label Ratio', loc='center right', bbox_to_anchor=(1.35,0.5))
  #   # -----

  #   plt.title(title)

  #   jpg_file = f"{group_col[0]}_{group_col[1]}_{dataset}_{str(seed)}.png"
  #   self.path_maker(self.path+dataset)
  #   plt.savefig(self.path+dataset+"/"+ jpg_file)
  #   plt.show()
  def double_var_pie_chart(self, group_col=["Label", "gender"], group_label_dic_var1={1: 'Toxic', 0: 'Non-Toxic'},
                          group_label_dic_var2={1: 'Gender', 0: 'Non-Gender'}, title="add title", seed=1000,
                          dataset="Noname", threshold=0):

    # Group the data by the Label and Gender columns
    grouped_data = self.data.groupby(group_col).size().reset_index(name='counts')

    # Replace the Label values with custom labels
    grouped_data[group_col[0]] = grouped_data[group_col[0]].replace(group_label_dic_var1)
    grouped_data[group_col[1]] = grouped_data[group_col[1]].replace(group_label_dic_var2)

    # Calculate the percentage for each slice
    total = grouped_data['counts'].sum()
    grouped_data['percentage'] = grouped_data['counts'] / total * 100

    # Filter out slices with a percentage less than the threshold
    if threshold > 0:
        filtered_data = grouped_data[grouped_data['percentage'] >= threshold]
    else:
        filtered_data = grouped_data

    # Plot the pie chart
    sns.set_style("darkgrid")
    patches,a = plt.pie(filtered_data['percentage'], startangle=90)
    plt.axis('equal')

    # Create a legend
    labels = [f"{label}: {percentage:.2f}%" for label, percentage in
              zip(filtered_data[group_col[0]] + ' \\ ' + filtered_data[group_col[1]].astype(str),
                  filtered_data['percentage'])]
    plt.legend(handles=patches, labels=labels, title='Label\\{} Ratio'.format(group_col[1].capitalize()), loc='center right', bbox_to_anchor=(1.39, 0.5), fontsize=11)

    # plt.title(title)

    jpg_file = f"{group_col[0]}_{group_col[1]}_{dataset}_{str(seed)}.png"
    self.path_maker(self.path + dataset)
    plt.savefig(self.path + dataset + "/" + jpg_file)
    plt.show()
 


  def fine_term(self, seed=1000, dataset="Noname", title="no title"):

    # # Get the counts of each fine term
    # list_of_fine_terms = [fine_term if row != 'None' else 'No Target' 
    #                       for row in self.data['attr'].dropna().tolist() 
    #                       for fine_term in row]
    # fine_terms = pd.Series(list_of_fine_terms).value_counts()

    # # Plot the horizontal bar chart
    # sns.set_style("darkgrid")
    # sns.set_palette("deep")
    # ax = sns.barplot(x=fine_terms.values, y=fine_terms.index, orient="h")

    # # Add the count on the right side of each bar
    # for i in range(len(fine_terms)):
    #     ax.text(fine_terms.values[i] + 10, i, str(fine_terms.values[i]), va='center')

    # # Add labels and title
    # plt.xlabel('Count')
    # plt.ylabel('Fine Term')
    # plt.title(title)

    # # Save and show the plot
    # jpg_file = f"Fine_term_{dataset}_{str(seed)}.png"
    # self.path_maker(self.path+dataset)
    # plt.savefig(self.path+dataset+"/" +jpg_file)
    # plt.show()

    # Get the counts of each fine term
    list_of_fine_terms = [fine_term if row != 'None' else 'No Target'                       for row in self.data['attr'].dropna().tolist() 
                          for fine_term in row]
    fine_terms = pd.Series(list_of_fine_terms).value_counts()

    # Convert the Series object to a DataFrame
    table_data = fine_terms.reset_index()
    table_data.columns = ['Fine Term', 'Count']

    # Display the table
    print(table_data)
    

  def fine_term_label(self,seed = 1000, dataset = "Noname", title = "no title"):


    # Get the count of each fine term for label 0 and label 1 separately
    fine_terms_0 = pd.Series([fine_term+" \\ "+"Non-Toxic" for row, label in zip(self.data['attr'].dropna().tolist(), self.data['Label'].dropna().tolist()) if label == 0 for fine_term in row]).value_counts()
    fine_terms_1 = pd.Series([fine_term +" \\ "+"Toxic" for row, label in zip(self.data['attr'].dropna().tolist(), self.data['Label'].dropna().tolist()) if label == 1 for fine_term in row]).value_counts()

    # Plot the two bar charts side by side
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    sns.set_style("darkgrid")
    sns.barplot(x=fine_terms_0.index, y=fine_terms_0.values, ax=ax[0])
    sns.barplot(x=fine_terms_1.index, y=fine_terms_1.values, ax=ax[1])

    # Add the count on top of each bar for label 0
    for i in range(len(fine_terms_0)):
        ax[0].text(i, fine_terms_0.values[i], str(fine_terms_0.values[i]), ha='center')


    # Add the count on top of each bar for label 1
    for i in range(len(fine_terms_1)):
        ax[1].text(i, fine_terms_1.values[i], str(fine_terms_1.values[i]), ha='center')


    ax[0].set_xlabel('Fine Term (Label=0)')
    ax[0].set_ylabel('Count')
    ax[0].set_xticklabels(fine_terms_0.index, rotation=90)

    ax[1].set_xlabel('Fine Term (Label=1)')
    ax[1].set_ylabel('Count')
    ax[1].set_xticklabels(fine_terms_1.index, rotation=90)

    fig.suptitle(title, y=1.05, fontsize=16)

    jpg_file = f"Fine_term_{dataset}_{str(seed)}_based on_thier_labels.png"
    self.path_maker(self.path+dataset)

    plt.savefig(self.path+dataset+"/"+jpg_file)

    plt.show()


  


  def plot_data_stat(self):

    no_toxic_no_protected = len(self.data.query('Label == 0 and `gender` == 0 and `race` == 0'))
    no_toxic_race_protected = len(self.data.query('Label == 0 and `gender` == 0 and `race` == 1'))
    no_toxic_gender_protected = len(self.data.query('Label == 0 and `gender` == 1 and `race` == 0'))
    no_toxic_both_protected = len(self.data.query('Label == 0 and `gender` == 1 and `race` == 1'))
    toxic_no_protected = len(self.data.query('Label == 1 and `gender` == 0 and `race` == 0'))
    toxic_race_protected = len(self.data.query('Label == 1 and `gender` == 0 and `race` == 1'))
    toxic_gender_protected = len(self.data.query('Label == 1 and `gender` == 1 and `race` == 0'))
    toxic_both_protected = len(self.data.query('Label == 1 and `gender` == 1 and `race` == 1'))
    
    categories = ['No Toxic\nNo Protected', 'No Toxic\nRace Protected', 'No Toxic\nGender Protected',  'No Toxic\nBoth Protected',
                  'Toxic\nNo Protected', 'Toxic\nRace Protected', 'Toxic\nGender Protected', 'Toxic\nBoth Protected']
    print("categories",len(categories))

    values = [no_toxic_no_protected, no_toxic_race_protected, no_toxic_gender_protected, no_toxic_both_protected,toxic_no_protected, 
              toxic_race_protected, toxic_gender_protected,toxic_both_protected]
    print("values",len(values))
    plt.figure(figsize=(10,5))
    ax = sns.barplot(x=categories, y=values, palette='Blues', edgecolor='w')
    for i in range(len(values)):
        ax.text(i, values[i], str(values[i]), ha='center')
        plt.xticks(rotation=90)


    plt.ylabel('Count')
    plt.xlabel('Groups')
    plt.title('Count of Data by SubGroups')
    plt.show()

  def plot_sampeled_data_stat(self):

    no_toxic_no_protected_unlabel = len(self.data.query('Label == 0 and `gender` == 0 and `race` == 0 and target ==0'))
    no_toxic_no_protected_label = len(self.data.query('Label == 0 and `gender` == 0 and `race` == 0 and target !=0'))

    no_toxic_race_protected_unlabel = len(self.data.query('Label == 0 and `gender` == 0 and `race` == 1 and target ==0'))
    no_toxic_race_protected_label = len(self.data.query('Label == 0 and `gender` == 0 and `race` == 1 and target !=0'))

    no_toxic_gender_protected_unlabel = len(self.data.query('Label == 0 and `gender` == 1 and `race` == 0 and target ==0'))
    no_toxic_gender_protected_label = len(self.data.query('Label == 0 and `gender` == 1 and `race` == 0 and target !=0 '))

    no_toxic_both_protected_unlabel = len(self.data.query('Label == 0 and `gender` == 1 and `race` == 1 and target ==0'))
    no_toxic_both_protected_label = len(self.data.query('Label == 0 and `gender` == 1 and `race` == 1 and target !=0'))

    toxic_no_protected_unlabel = len(self.data.query('Label == 1 and `gender` == 0 and `race` == 0 and target ==0'))
    toxic_no_protected_label = len(self.data.query('Label == 1 and `gender` == 0 and `race` == 0 and target !=0'))

    toxic_race_protected_unlabel = len(self.data.query('Label == 1 and `gender` == 0 and `race` == 1 and target == 0'))
    toxic_race_protected_label = len(self.data.query('Label == 1 and `gender` == 0 and `race` == 1 and target !=0 '))

    toxic_gender_protected_unlabel = len(self.data.query('Label == 1 and `gender` == 1 and `race` == 0 and target ==0'))
    toxic_gender_protected_label = len(self.data.query('Label == 1 and `gender` == 1 and `race` == 0 and target !=0 '))

    toxic_both_protected_unlabel = len(self.data.query('Label == 1 and `gender` == 1 and `race` == 1 and target == 0'))
    toxic_both_protected_label = len(self.data.query('Label == 1 and `gender` == 1 and `race` == 1 and target !=0'))

    
    categories = ['No Toxic\ No Protected\ Unlabel','No Toxic\ No Protected\ label', 'No Toxic\ Race Protected\ Unlabel','No Toxic\ Race Protected\ label', 'No Toxic\ Gender Protected\ Unlabel', 'No Toxic\ Gender Protected\ label', 
                  'No Toxic\ Both Protected\ Unlabel','No Toxic\ Both Protected\ label',
                  'Toxic\ No Protected\ Unlabel','Toxic\ No Protected\ label', 'Toxic\ Race Protected\ Unlabel','Toxic\ Race Protected\ label', 'Toxic\ Gender Protected\ Unlabel', 'Toxic\ Gender Protected\ label',
                  'Toxic\ Both Protected\ Unlabel','Toxic\ Both Protected\ label']
    print("categories",len(categories))
    values = [ no_toxic_no_protected_unlabel, no_toxic_no_protected_label, no_toxic_race_protected_unlabel, no_toxic_race_protected_label, 
              no_toxic_gender_protected_unlabel, no_toxic_gender_protected_label,no_toxic_both_protected_unlabel,no_toxic_both_protected_label,
              toxic_no_protected_unlabel,toxic_no_protected_label,toxic_race_protected_unlabel,toxic_race_protected_label,toxic_gender_protected_unlabel,toxic_gender_protected_label,
              toxic_both_protected_unlabel,toxic_both_protected_label]

    short_labels = ['NTNP_U','NTNP_L', 'NTR_U','NTR_L', 'NTG_U', 'NTG_L',
'NTBP_U','NTBP_L','TNP_U','TNP_L', 'TR_U','TR_L', 'TG_U', 'TG_L',
'TBP_U','TBP_L',]

    print("values",len(values))

    plt.figure(figsize=(10,5))
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    ax = sns.barplot(x=short_labels, y=values, palette='Blues', edgecolor='w', ax=axs[0])
    for i in range(len(values)):
        ax.text(i, values[i], str(values[i]), ha='center')
    ax.set_xticklabels(short_labels, rotation=90)
    ax.set_ylabel('Count')
    ax.set_xlabel('Groups')
    ax.set_title('Count of Data by Groups')
    
    mapping_df = pd.DataFrame({'Short Label': short_labels, 'Category': categories})
    table = axs[1].table(cellText=mapping_df.values, colLabels=mapping_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    axs[1].axis('tight')
    axs[1].axis('off')
    # axs[1].set_title('Legend')
    
    plt.show()

  def plot_generator(self,seed = 1000, dataset = "Noname"):

     self.single_var_pie_chart(group_col = "Label", group_label_dic ={1: 'Toxic', 0: 'Non-Toxic'}, title = "Label Distribution in {}".format(dataset),seed = seed, dataset = dataset)
     self.single_var_pie_chart(group_col = "gender", group_label_dic ={1: 'Gender', 0: 'Non-Gender'}, title = "Gender Distribution in {}".format(dataset),seed = seed, dataset = dataset)
     self.single_var_pie_chart(group_col = "race", group_label_dic ={1: 'Race', 0: 'Non-Race'}, title = "Race Distribution in {}".format(dataset),seed = seed, dataset = dataset)

     self.double_var_pie_chart(group_col = ["Label","gender"],  group_label_dic_var1 ={1: 'Toxic', 0: 'Non-Toxic'},
                           group_label_dic_var2 ={1: 'Gender', 0: 'Non-Gender'}, title = "Label Gender Distribution in {}".format(dataset),seed = seed, dataset = dataset)
     
     self.double_var_pie_chart(group_col = ["Label","race"],  group_label_dic_var1 ={1: 'Toxic', 0: 'Non-Toxic'},
                           group_label_dic_var2 ={1: 'Race', 0: 'Non-Race'}, title = "Label Race Distribution in {}".format(dataset),seed = seed, dataset = dataset )
      
     self.fine_term(seed = seed, dataset = dataset , title = "Fine Term Distribution in {}".format(dataset))

     self.fine_term_label(seed = seed, dataset = dataset , title = "Fine Term Distribution over Labels in {}".format(dataset))
     self.plot_data_stat()
     self.plot_sampeled_data_stat()

  def make_autopct(self,values):
      def my_autopct(pct):
          total = sum(values)
          val = int(round(pct*total/100.0))
          return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
      return my_autopct




'''
class TextClassificationModel
class DebiasingMethod
class TrainingLog
class PlottingUtils
class Plot_data
'''


# !pip install transformers==4.3.2
# !pip install transformers==4.26.0



# import torch
# import io
# import random
# import time
# import math
# import datetime
# import warnings
# warnings.filterwarnings('ignore')
# import logging
# logging.basicConfig(level=logging.ERROR)
# import ast
# import os
# 
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
# from collections import defaultdict
# 
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from pylab import rcParams
# import matplotlib.pyplot as plt
# from matplotlib import rc
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,precision_score,recall_score,f1_score
# from sklearn.metrics import matthews_corrcoef
# # import sentencepiece
# 
# import torch.nn.functional as F
# import torch.nn as nn
# from transformers import *
# 
# #!pip install sentencepiece
# from sklearn import metrics
# 
# 
# 
# import sys
# sys.path.append('/content/drive/MyDrive/SS_Fair/Metrics_Evaluations/')
# import metrics_eval
# 
# # address
# import sys
# sys.path.append('/content/drive/MyDrive/SS_Fair/Models')
# import Arch
# 
# import sys
# sys.path.append('/content/drive/MyDrive/SS_Fair/Dataloaders/')
# import data_loaders
# import Data_sampler
# 
# 
# import shutil
# thres = 0.5
# 
# # If there's a GPU available...
# if torch.cuda.is_available():
#     # Tell PyTorch to use the GPU.
#     device = torch.device("cuda")
#     print('There are %d GPU(s) available.' % torch.cuda.device_count())
#     print('We will use the GPU:', torch.cuda.get_device_name(0))
# # If not...
# else:
#     print('No GPU available, using the CPU instead.')
#     device = torch.device("cpu")
# 
# 
# from prettytable import PrettyTable
# import csv
# from google.colab import files
# import itertools
# 
# from matplotlib.pylab import True_
# 
# from pandas.core.describe import describe_timestamp_as_categorical_1d
# import seaborn as sns
# 
# 
# import seaborn as sns
# 
# class TextClassificationModel:
# 
#     def __init__(self,log_training,plot_training):
#         '''
#         it comes from the FairTextClassification
#         '''
# 
#         self.log_training = log_training
#         self.plot_training = plot_training
# 
# 
#     def performance_metrics(self,data_dict,epoch,mode):
#       print("11111111111111111",data_dict["predictions_net"])
#       performance_metrics = metrics_eval.Performance_Metrics.get_metrics(data_dict["truths"], data_dict["predictions_net"],data_dict['identities_gender'],data_dict['identities_race'])
#       try: 
#         fpr, tpr, _ = metrics.roc_curve(data_dict["truths"], data_dict["y_scores"], pos_label=1)
#       except:
#         print(len(data_dict["truths"]), len(data_dict["y_scores"]))
# 
#       roc_auc = metrics.auc(fpr, tpr)
#       confusion_matrix = metrics.confusion_matrix(data_dict["truths"], data_dict["predictions_net"])
#       metrics_dic_val_gender = metrics_eval.Metrics(data_dict["predictions_net"], data_dict["truths"], data_dict["identities_gender"], thres).all_metrics("gender")
#       metrics_dic_val_race = metrics_eval.Metrics(data_dict["predictions_net"], data_dict["truths"], data_dict["identities_race"], thres).all_metrics("race")
# 
#       performance_log = {**performance_metrics, **metrics_dic_val_gender, **metrics_dic_val_race}
#       performance_log["roc_auc"] = roc_auc
#       performance_log["confusion_matrix"] = confusion_matrix
#       performance_log["Mode"] = mode
# 
# 
#       self.print_performance_log(performance_log)
# 
#       # object of TrainingLog
#       self.log_training.update_epoch_log( performance_log, epoch)
#       '''
#       You must defined a object of PrettyTable as you initialized your model in FairTextClassification
#       '''
#       return performance_log
# 
#     def performance_metrics_fine_term(self, data_dict, epoch,fine_terms_list ,mode):
# 
#       predictions_net = data_dict['predictions_net']
#       truths = data_dict['truths']
#       identities_gender = data_dict['identities_gender']
#       identities_race = data_dict['identities_race']
#       fine_terms = data_dict['fine_terms']
# 
#       print("finter_function",fine_terms_list)
#       # fine_terms_list = ['gender', 'race']
# 
#       # race_term = set(["African", "Arab", "Asian", "Caucasian", "Hispanic","Refugee"])
#       # religion_term =  set(["Islam", "Buddhism", "Jewish","Hindu", "Christian"])
#       # gender_term =  set(["Men", "Women","Homosexual"])
# 
#       # fine_terms_list = [gender_term,race_term]
#       thres = 0.5
# 
#       # Initialize dictionary to store performance metrics for each fine-grained term
#       fine_term_metrics = {'gender': {}, 'race': {}}
# 
#       # Compute performance metrics for each fine-grained term
#       fine_gender_metrics = metrics_eval.Metrics_fine_terms(predictions_net, truths, identities_gender, fine_terms, fine_terms_list[0], data_dict['y_scores'],thres).all_metrics_terms()
#       fine_race_metrics = metrics_eval.Metrics_fine_terms(predictions_net, truths, identities_race, fine_terms, fine_terms_list[1], data_dict['y_scores'],thres).all_metrics_terms()
# 
#       # Add performance metrics to dictionary
#       fine_term_metrics['gender'] = fine_gender_metrics
#       fine_term_metrics['race'] = fine_race_metrics
#       fine_term_metrics["Mode"] = mode
# 
#       self.print_performance_log(fine_term_metrics)
# 
#       # object of TrainingLog
#       self.log_training.update_epoch_log( fine_term_metrics, epoch, True)
# 
#       return fine_term_metrics
# 
# 
#     def save_log_epoch(self):
# 
#        self.log_training
# 
#     '''
#     must go to print class
#     '''
#     def print_performance_log(self,performance_log):
# 
#         table = PrettyTable()
#         table.field_names = []
# 
#         # Add the performance metrics
#         for key, value in performance_log.items():
#             table.add_row([key, value])
# 
#         print(table)
# 
# 
# 
# 
# class DebiasingMethod:
# 
#     def __init__(self,log_training,plot_training):
#         '''
#         it comes from the FairTextClassification
#         '''
# 
#         self.log_training = log_training
#         self.plot_training = plot_training
# 
# 
#     def selection_score_computer(self,best_accuracy ,max_shift_acc = 0.07):
# 
#       scaled_max_shift_acc = max_shift_acc*best_accuracy
# 
#       return scaled_max_shift_acc
# 
# 
# 
# 
#     def performance_metrics(self,data_dict,epoch,mode, save = True):
# 
#       performance_metrics = metrics_eval.Performance_Metrics.get_metrics(data_dict["truths"], data_dict["predictions_net"],data_dict['identities_gender'],data_dict['identities_race'])
#       try: 
#         fpr, tpr, _ = metrics.roc_curve(data_dict["truths"], data_dict["y_scores"], pos_label=1)
#       except:
#         print(len(data_dict["truths"]), len(data_dict["y_scores"]))
# 
#       roc_auc = metrics.auc(fpr, tpr)
#       confusion_matrix = metrics.confusion_matrix(data_dict["truths"], data_dict["predictions_net"])
#       metrics_dic_val_gender = metrics_eval.Metrics(data_dict["predictions_net"], data_dict["truths"], data_dict["identities_gender"], thres).all_metrics("gender")
#       metrics_dic_val_race = metrics_eval.Metrics(data_dict["predictions_net"], data_dict["truths"], data_dict["identities_race"], thres).all_metrics("race")
# 
#       performance_log = {**performance_metrics, **metrics_dic_val_gender, **metrics_dic_val_race}
#       performance_log["roc_auc"] = roc_auc
#       performance_log["confusion_matrix"] = confusion_matrix
#       performance_log["Mode"] = mode
# 
# 
#       self.print_performance_log(performance_log)
# 
#       # object of TrainingLog
#       if save == True:
#         self.log_training.update_epoch_log( performance_log, epoch)
# 
#       return performance_log
# 
#     def performance_metrics_fine_term(self, data_dict, epoch,fine_terms_list, mode):
# 
#       predictions_net = data_dict['predictions_net']
#       truths = data_dict['truths']
#       identities_gender = data_dict['identities_gender']
#       identities_race = data_dict['identities_race']
#       fine_terms = data_dict['fine_terms']
# 
#       print("finter_function",fine_terms_list)
#       print("finterm0",data_dict['fine_terms'])
# 
#       fine_terms_list = ['gender', 'race']
#       race_term = set(["African", "Arab", "Asian", "Caucasian", "Hispanic","Refugee"])
#       religion_term =  set(["Islam", "Buddhism", "Jewish","Hindu", "Christian"])
#       gender_term =  set(["Men", "Women","Homosexual"])
# 
#       fine_terms_list = [gender_term,race_term]
#       print("finter_inside",fine_terms_list)
# 
# 
#       thres = 0.5
# 
#       # Initialize dictionary to store performance metrics for each fine-grained term
#       fine_term_metrics = {'gender': {}, 'race': {}}
# 
#       # Compute performance metrics for each fine-grained term
#       fine_gender_metrics = metrics_eval.Metrics_fine_terms(predictions_net, truths, identities_gender, fine_terms, fine_terms_list[0], data_dict['y_scores'],thres).all_metrics_terms()
#       fine_race_metrics = metrics_eval.Metrics_fine_terms(predictions_net, truths, identities_race, fine_terms, fine_terms_list[1], data_dict['y_scores'],thres).all_metrics_terms()
# 
#       # Add performance metrics to dictionary
#       fine_term_metrics['gender'] = fine_gender_metrics
#       fine_term_metrics['race'] = fine_race_metrics
#       fine_term_metrics["Mode"] = mode
# 
#       self.print_performance_log(fine_term_metrics)
# 
#       # object of TrainingLog
#       self.log_training.update_epoch_log( fine_term_metrics, epoch, True)
# 
#       return fine_term_metrics
# 
# 
#     # def save_log_epoch(self):
# 
#     #    self.log_training
# 
#     # '''
#     # must go to print class
#     # '''
#     def print_performance_log(self,performance_log):
# 
#         table = PrettyTable()
#         table.field_names = []
# 
#         # Add the performance metrics
#         for key, value in performance_log.items():
#             table.add_row([key, value])
# 
#         # table.set_row_style(1, "background-color: yellow")
# 
#         print(table)
# 
# 
# 
#     def remove_model(self,path):
#       if os.path.isfile(path):
#           os.remove(path)
#       elif os.path.isdir(path):
#           shutil.rmtree(path)
#       else:
#           raise ValueError(f"Invalid path {path}")
# 
# 
# 
# 
# 
# class TrainingLog:
# 
#     def __init__(self, model_name, seed, num_label_data, num_unlabel_data):
#         self.model_name = model_name
#         self.seed = seed
#         self.num_label_data = num_label_data
#         self.num_unlabel_data = num_unlabel_data
#         '''
#         you probably need to change it in a way that it is a dic (seed) of dic(epoch) of dic(metrics)
#         '''
#         self.epoch_logs = {}
#         self.epoch_logs_fine_terms = {}
# 
#         '''
#         add each seed best model dictionary, a dic (seed) of a dic (best model)
#         '''
#         self.best_model = {}
# 
#         self.epoch_coarse = {}
#         self.epoch_fine = {}
# 
# 
#     '''
#     all must change to include seed, also add a function to update the best model
#     '''    
# 
#     def update_epoch_log(self, data, epoch, fine_term = False):
# 
#        if fine_term == True:
# 
#           self.epoch_fine[epoch] = data
# 
# 
#           table = self.epoch_logs_fine_terms.get(epoch, None)
#           if table is None:
#               table = PrettyTable()
#               table.field_names = []
#               self.epoch_logs_fine_terms[epoch] = table
# 
#           # Add the columns using the keys of the dictionary
#           if not table.field_names:
#               table.field_names = list(data.keys())
# 
#           # Add the values as a row
#           table.add_row(list(data.values()))
# 
#        else:
#           self.epoch_coarse[epoch] = data
#           table = self.epoch_logs.get(epoch, None)
#           if table is None:
#               table = PrettyTable()
#               table.field_names = []
#               self.epoch_logs[epoch] = table
# 
#           # Add the columns using the keys of the dictionary
#           if not table.field_names:
#               table.field_names = list(data.keys())
# 
#           # Add the values as a row
#           table.add_row(list(data.values()))
# 
# 
#     def get_epoch_coarse(self):
#       return self.epoch_coarse
# 
# 
#     def get_epoch_fine(self):
#       return self.epoch_fine
# 
# 
# 
# 
# 
# 
#     def save_log(self, path,fine_term= False):
# 
#       csv_file = f"{self.model_name}_seed{self.seed}_label{self.num_label_data}_unlabel{self.num_unlabel_data}.csv"
# 
# 
#       if fine_term == False:
# 
#         epoch = list( self.epoch_coarse.keys())
#         # # print("epoch",epoch)
# 
#         # extract the column names (keys of the inner dictionaries)
#         columns = list( self.epoch_coarse[epoch[0]].keys())
#         # print("columns",columns)
# 
#         data = {col: [self.epoch_coarse[epoch[i]][col] for i in range(len(epoch))] for col in columns}
#         print("data",data)
# 
#         df = pd.DataFrame(data, columns=columns)
# 
#         # Add the 'Epoch' column
#         df.insert(0, 'Epoch', range(1, len(epoch)+1))
# 
#         # Write the dataframe to a CSV file
#         df.to_csv(path+csv_file, index=False)
# 
# 
# 
# 
#       else:
# 
# 
# 
#         # Create an empty dataframe
#         df = pd.DataFrame()
# 
#         # Get the keys of the outer dictionary (0 or 1)
#         epoch = list(self.epoch_fine.keys())
# 
#         # Extract the column names (keys of the inner dictionaries) for the 'gender' key
#         columns_gender = self.epoch_fine[epoch[0]]['gender']
#         prefix_columns_gen = [col for col in columns_gender.keys()]
#         suffix_col_gen = [col for col in list(columns_gender.values())[0].keys()]
#         gender_cols = [x+"_"+y for x, y in itertools.product(prefix_columns_gen, suffix_col_gen)]
# 
#         # Extract the column names (keys of the inner dictionaries) for the 'race' key
#         columns_race = self.epoch_fine[epoch[0]]['race']
#         prefix_columns_race = [col for col in columns_race.keys()]
#         suffix_col_race = [col for col in list(columns_race.values())[0].keys()]
#         race_cols = [x+"_"+y for x, y in itertools.product(prefix_columns_race, suffix_col_race)]
# 
#         # Combine the column names for 'gender' and 'race'
#         all_cols = gender_cols + race_cols
# 
#         # Add a 'Mode' column
#         all_cols.append('Mode')
# 
#         # Add the key values (0 or 1) as the first column in the dataframe
#         df['epoch'] = epoch
#         cc = 0
#         dd= 0
#         # Iterate over the outer dictionary and add the values for each key to the dataframe
#         for e in epoch:
#             for g in prefix_columns_gen:
#                 for c in suffix_col_gen:
# 
#                     try:
#                       df["gender_"+g+"_"+c] = self.epoch_fine[e]['gender'][g][c]
#                     #   dd +=1 
#                     except:
#                     #   cc +=1
#                       pass
# 
# 
# 
#             for r in prefix_columns_race:
#                 for c in suffix_col_race:
#                     try:
#                       df["race_"+r+"_"+c] = self.epoch_fine[e]['race'][r][c]
# 
#                     except:
#                       pass
# 
# 
#             df['Mode'] = self.epoch_fine[e]['Mode']
# 
#         # Save the dataframe to a csv file
#         df.to_csv(path+"fine_"+csv_file, index=False)
# 
# 
#     def save_best_log(self, path,best_dic,fine_term= False):
# 
#       csv_file = f"BEST_{self.model_name}_seed{self.seed}_label{self.num_label_data}_unlabel{self.num_unlabel_data}.csv"
# 
# 
#       if fine_term == False:
# 
#         # epoch = list( self.epoch_coarse.keys())
#         # # print("epoch",epoch)
# 
#         # extract the column names (keys of the inner dictionaries)
#         columns = list( best_dic.keys())
#         # print("columns",columns)
# 
#         data = {col: [best_dic[col]] for col in columns}
#         print("best",data)
# 
#         df = pd.DataFrame(data, columns=columns)
# 
#         print("best df",df)
# 
#         # # Add the 'Epoch' column
#         # df.insert(0, 'Epoch', range(1, len(epoch)+1))
# 
#         # Write the dataframe to a CSV file
#         print("path",path+csv_file)
#         df.to_csv(path+csv_file, index=False)
# 
# 
# 
# 
#       else:
# 
#         print("best_fine",best_dic)
# 
# 
#         # Create an empty dataframe
#         df = pd.DataFrame()
# 
#         # Get the keys of the outer dictionary (0 or 1)
#         # epoch = list(self.epoch_fine.keys())
# 
#         # Extract the column names (keys of the inner dictionaries) for the 'gender' key
#         columns_gender = best_dic['gender']
#         # print("columns_gender",columns_gender)
#         prefix_columns_gen = [col for col in columns_gender.keys()]
#         # print("prefix_columns_gen",prefix_columns_gen)
#         suffix_col_gen = [col for col in list(columns_gender.values())[0].keys()]
#         # print("suffix_col_gen",suffix_col_gen)
# 
#         gender_cols = [x+"_"+y for x, y in itertools.product(prefix_columns_gen, suffix_col_gen)]
# 
#         # Extract the column names (keys of the inner dictionaries) for the 'race' key
#         columns_race = best_dic['race']
#         prefix_columns_race = [col for col in columns_race.keys()]
#         suffix_col_race = [col for col in list(columns_race.values())[0].keys()]
#         race_cols = [x+"_"+y for x, y in itertools.product(prefix_columns_race, suffix_col_race)]
# 
#         # Combine the column names for 'gender' and 'race'
#         all_cols = gender_cols + race_cols
# 
#         # Add a 'Mode' column
#         all_cols.append('Mode')
# 
#         # Add the key values (0 or 1) as the first column in the dataframe
#         # df['epoch'] = epoch
#         cc = 0
#         dd= 0
#         # Iterate over the outer dictionary and add the values for each key to the dataframe
#         # for e in epoch:
#         for g in prefix_columns_gen:
#                 for c in suffix_col_gen:
# 
#                     try:
#                       # print(" best_dic['gender'][g][c]", best_dic['gender'][g][c])
#                       df["gender_"+g+"_"+c] = [best_dic['gender'][g][c]]
#                     #   dd +=1 
#                     except:
#                     #   cc +=1
#                       print("error", g,c)
#                       pass
# 
#         print("0 df",df)
# 
# 
#         for r in prefix_columns_race:
#                 for c in suffix_col_race:
#                     try:
#                       df["race_"+r+"_"+c] = [best_dic['race'][r][c]]
# 
#                     except:
#                       pass
# 
# 
#         df['Mode'] = best_dic['Mode']
# 
#         # print("best df",df)
# 
# 
#         # Save the dataframe to a csv file
#         df.to_csv(path+"fine_"+csv_file, index=False)
# 
# 
# 
# # PlottingUtils class contains methods for plotting the results
# class PlottingUtils:
# 
#   def __init__(self,path,model_name,seed,num_label_data,num_unlabel_data):
#      self.path = path
#      self.model_name = model_name
#      self.seed = seed
#      self.num_label_data = num_label_data
#      self.num_unlabel_data = num_unlabel_data
# 
# 
#   def plot_iteration_score(self, fairness_metrics_iteration, metrics, best_epochs):
# 
#     jpg_file = f"{self.model_name}_seed{self.seed}_label{self.num_label_data}_unlabel{self.num_unlabel_data}_{metrics}.png"
#     best_scores =[]
#     # Create a dataframe with the metric scores and epochs
#     data = {'epoch': range(1,len(fairness_metrics_iteration.keys())+1)}
#     for i, metric in enumerate(metrics):
#         data[metric] = [float(fairness_metrics_iteration[epoch][metric]) for epoch in fairness_metrics_iteration  for score in fairness_metrics_iteration[epoch] if score == metric ]
#         best_scores.append(data[metric][best_epochs-1])
# 
#     df = pd.DataFrame(data)
#     # Plot the data using Seaborn
#     sns.set_style("darkgrid")
#     plt.figure(figsize=(10,5))
#     colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan', 'magenta']
#     for i, metric in enumerate(metrics):
#         sns.scatterplot(x=[best_epochs], y=[best_scores[i]], color='black', marker='*',s=250,zorder=3) #[best_score] plot the best_epoch
#         separated = metric.split("_", 1)
#         if len(separated) !=1:
#           metric_formal = separated[0].capitalize() +self.title_dic(separated[1])
#         else: 
#           metric_formal = metric
#         sns.lineplot(x='epoch', y=metric, data=df, color=colors[i], marker='o', markersize=6, label=metric_formal,zorder=1)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# 
#     plt.xlabel('Epoch')
#     plt.ylabel('Metrics')
#     plt.title("Model performance per epoch")
#     plt.savefig(self.path +jpg_file, bbox_inches='tight', dpi=300)
#     plt.show()
# 
#   def plot_iteration_score_fine_grain(self, fairness_metrics_iterations, metric, best_epochs,acc_separation = None, type_bias= ""):
# 
#       jpg_file = f"Fine_{self.model_name}_seed{self.seed}_label{self.num_label_data}_unlabel{self.num_unlabel_data}_{metric}.png"
# 
#       color = ['black','red','orange', "blue","green","cyan","magenta","yellow","black"]
#       metric_score_list = defaultdict(list)
#       stats = []
#       fine_terms = []
#       print("fairness_metrics_iterations",fairness_metrics_iterations)
#       for idx,epoch in enumerate(fairness_metrics_iterations):
#           for key1 in fairness_metrics_iterations[epoch]: #iterate over first level keys (epochs)
# 
#               if key1 != 'Mode' and acc_separation == None:
#                 pass
# 
#               elif key1 != 'Mode' and acc_separation != key1:
#                 continue
# 
# 
#               elif key1 != 'Mode' and acc_separation == key1:
#                 pass
# 
#               else:
#                 continue
# 
# 
#               for key2 in fairness_metrics_iterations[epoch][key1]: #iterate over second level keys (gender, race)
# 
#                             fine_term = key1+"*"+ key2 + '*' + metric
#                             if fine_term not in fine_terms:
# 
#                                 fine_terms.append(fine_term)
#                                 # print(key1,key2,metric)
#                                 metric_score_list[fine_term].append(fairness_metrics_iterations[epoch][key1][key2][metric])
#                                 stats.append(fairness_metrics_iterations[epoch][key1][key2]['number of samples'])
#                             else:
# 
#                                 metric_score_list[fine_term].append(fairness_metrics_iterations[epoch][key1][key2][metric])
#                                 stats.append(fairness_metrics_iterations[epoch][key1][key2]['number of samples'])
# 
# 
# 
#       data = []
#       print("metric_score_list",metric_score_list)
#       data = pd.DataFrame(metric_score_list)
#       data['epoch'] = [i+1 for i, x in enumerate(data.index)]
# 
#       for subgroup in fine_terms:
#         subgroup_label = subgroup.split('*')[0].capitalize()+" " + subgroup.split('*')[1].capitalize() + self.title_dic(subgroup.split('*')[2])
#         subgroup_data = data[[subgroup, 'epoch']]
# 
#         sns.lineplot(x='epoch', y=subgroup, data=subgroup_data, label=subgroup_label)
#         sns.scatterplot(x=[best_epochs], y=[data[subgroup][best_epochs-1]], color='black', marker='*',s=250,zorder=3) #[best_score] plot the best_epoch
# 
#       label=subgroup
# 
#       print("subgroup",subgroup)
#       plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# 
# 
#       plt.ylabel(self.title_dic(metric)) #set the label for y axis
#       plt.xlabel('Epochs') #set the label for x-axis
#       plt.title("{} Performance Per epoch{}".format(self.title_dic(metric),type_bias)) #set the title of the graph
#       plt.savefig(self.path +jpg_file, bbox_inches='tight', dpi=300)
#       plt.show() #display the graph
# 
#   def title_dic(self,term):
# 
# 
#         term_dic ={"p_value":" P_value","demo_parity":" Demographic Parity","equ_odds":" Equalized Odds Difference","equ_odds_percent":" Equalized Odds Difference Percentage"
#         ,"equ_opportunity":" Equalized Opportunity Difference","equ_opportunity_percent":" Equalized Opportunity Difference Percentage"}
# 
#         return term_dic.get(term,term)
# 
# 
# 
# 
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# 
# class Plot_data:
# 
#   def __init__(self,data,path):
# 
#     self.data =data
#     self.path = path
#     self.data['attr'] = data['attr'].apply(ast.literal_eval)
# 
# 
#   def path_maker(self,folder_path):
# 
#       # Check if the folder exists
#       if not os.path.exists(folder_path):
#         # Create a new folder
#         # !mkdir $folder_path
#         os.makedirs(folder_path)
#         print("Folder created: ", folder_path)
#       else:
#         print("Folder already exists: ", folder_path)
# 
# 
#   def single_var_pie_chart(self,group_col = "Label", group_label_dic ={1: 'Toxic', 0: 'Non-Toxic'}, title = "add title",seed = 1000, dataset = "Noname"):
# 
#     # Group the data by the Label column
#     grouped_data = self.data.groupby(group_col).size().reset_index(name='counts')
#     # Replace the Label values with custom labels
#     grouped_data[group_col] = grouped_data[group_col].replace(group_label_dic)
# 
#     # Plot the pie chart
#     sns.set_style("darkgrid")
#     plt.pie(grouped_data['counts'], labels=grouped_data[group_col], startangle=90, autopct=self.make_autopct(grouped_data['counts']))
#     plt.axis('equal')
#     # plt.legend(title=group_col, labels=[list(group_label_dic.keys())[0], list(group_label_dic.keys())[0]], loc='upper left')
#     # -----
#     plt.title(title)
# 
#     jpg_file = f"{group_col}_{dataset}_{str(seed)}.png"
#     self.path_maker(self.path+dataset)
#     plt.savefig(self.path+dataset+"/"+ jpg_file)
# 
#     plt.show()
# 
#   def double_var_pie_chart(self,group_col = ["Label","gender"],  group_label_dic_var1 ={1: 'Toxic', 0: 'Non-Toxic'},
#                            group_label_dic_var2 ={1: 'Gender', 0: 'Non-Gender'}, title = "add title" ,seed = 1000, dataset = "Noname"):
# 
#     # Group the data by the Label and Gender columns
#     grouped_data = self.data.groupby(group_col).size().reset_index(name='counts')
# 
#     # Replace the Label values with custom labels
#     grouped_data[group_col[0]] = grouped_data[group_col[0]].replace(group_label_dic_var1)
#     grouped_data[group_col[1]] = grouped_data[group_col[1]].replace(group_label_dic_var2)
# 
#     # Plot the pie chart
#     sns.set_style("darkgrid")
#     plt.pie(grouped_data['counts'], labels=grouped_data[group_col[0]] + " \\ " + grouped_data[group_col[1]].astype(str), startangle=90, autopct=self.make_autopct(grouped_data['counts']))
#     plt.axis('equal')
#     # plt.legend(title='Gender keyword\Label Ratio', loc='center right', bbox_to_anchor=(1.35,0.5))
#     # -----
# 
#     plt.title(title)
# 
#     jpg_file = f"{group_col[0]}_{group_col[1]}_{dataset}_{str(seed)}.png"
#     self.path_maker(self.path+dataset)
#     plt.savefig(self.path+dataset+"/"+ jpg_file)
#     plt.show()
# 
# 
#   def fine_term(self,seed = 1000, dataset = "Noname", title = "no title"):
# 
# 
#     list_of_fine_terms = [fine_term if row != 'None' else 'No Target' 
#                           for row in self.data['attr'].dropna().tolist() 
#                           for fine_term in row]
#     fine_terms = pd.Series(list_of_fine_terms).value_counts()
#     print("fine_terms",fine_terms)
#     print("fine_terms.index",fine_terms.index)
#     print("fine_terms.values",fine_terms.values)
#     # Plot the bar chart
#     sns.set_style("darkgrid")
#     ax = sns.barplot(x=fine_terms.index, y=fine_terms.values)
# 
#     # Add the count on top of each bar
#     for i in range(len(fine_terms)):
#         ax.text(i, fine_terms.values[i], str(fine_terms.values[i]), ha='center')
# 
# 
#     plt.xlabel('Fine Term')
#     plt.ylabel('Count')
#     plt.xticks(rotation=45)
#     plt.title(title)
# 
#     jpg_file = f"Fine_term_{dataset}_{str(seed)}.png"
#     self.path_maker(self.path+dataset)
# 
#     plt.savefig(self.path+dataset+"/" +jpg_file)
# 
#     plt.show()
# 
#   def fine_term_label(self,seed = 1000, dataset = "Noname", title = "no title"):
# 
# 
#     # Get the count of each fine term for label 0 and label 1 separately
#     fine_terms_0 = pd.Series([fine_term+" \\ "+"Non-Toxic" for row, label in zip(self.data['attr'].dropna().tolist(), self.data['Label'].dropna().tolist()) if label == 0 for fine_term in row]).value_counts()
#     fine_terms_1 = pd.Series([fine_term +" \\ "+"Toxic" for row, label in zip(self.data['attr'].dropna().tolist(), self.data['Label'].dropna().tolist()) if label == 1 for fine_term in row]).value_counts()
# 
#     # Plot the two bar charts side by side
#     fig, ax = plt.subplots(1, 2, figsize=(15, 5))
# 
#     sns.set_style("darkgrid")
#     sns.barplot(x=fine_terms_0.index, y=fine_terms_0.values, ax=ax[0])
#     sns.barplot(x=fine_terms_1.index, y=fine_terms_1.values, ax=ax[1])
# 
#     # Add the count on top of each bar for label 0
#     for i in range(len(fine_terms_0)):
#         ax[0].text(i, fine_terms_0.values[i], str(fine_terms_0.values[i]), ha='center')
# 
#     # Add the count on top of each bar for label 1
#     for i in range(len(fine_terms_1)):
#         ax[1].text(i, fine_terms_1.values[i], str(fine_terms_1.values[i]), ha='center')
# 
#     ax[0].set_xlabel('Fine Term (Label=0)')
#     ax[0].set_ylabel('Count')
#     ax[0].set_xticklabels(fine_terms_0.index, rotation=45)
# 
#     ax[1].set_xlabel('Fine Term (Label=1)')
#     ax[1].set_ylabel('Count')
#     ax[1].set_xticklabels(fine_terms_1.index, rotation=45)
# 
#     fig.suptitle(title, y=1.05, fontsize=16)
# 
#     jpg_file = f"Fine_term_{dataset}_{str(seed)}_based on_thier_labels.png"
#     self.path_maker(self.path+dataset)
# 
#     plt.savefig(self.path+dataset+"/"+jpg_file)
# 
#     plt.show()
# 
# 
# 
# 
# 
#   def plot_data_stat(self):
# 
#     no_toxic_no_protected = len(self.data.query('Label == 0 and `gender` == 0 and `race` == 0'))
#     no_toxic_race_protected = len(self.data.query('Label == 0 and `gender` == 0 and `race` == 1'))
#     no_toxic_gender_protected = len(self.data.query('Label == 0 and `gender` == 1 and `race` == 0'))
#     no_toxic_both_protected = len(self.data.query('Label == 0 and `gender` == 1 and `race` == 1'))
#     toxic_no_protected = len(self.data.query('Label == 1 and `gender` == 0 and `race` == 0'))
#     toxic_race_protected = len(self.data.query('Label == 1 and `gender` == 0 and `race` == 1'))
#     toxic_gender_protected = len(self.data.query('Label == 1 and `gender` == 1 and `race` == 0'))
#     toxic_both_protected = len(self.data.query('Label == 1 and `gender` == 1 and `race` == 1'))
# 
#     categories = ['No Toxic\nNo Protected', 'No Toxic\nRace Protected', 'No Toxic\nGender Protected',  'No Toxic\nBoth Protected',
#                   'Toxic\nNo Protected', 'Toxic\nRace Protected', 'Toxic\nGender Protected', 'Toxic\nBoth Protected']
#     print("categories",len(categories))
# 
#     values = [no_toxic_no_protected, no_toxic_race_protected, no_toxic_gender_protected, no_toxic_both_protected,toxic_no_protected, 
#               toxic_race_protected, toxic_gender_protected,toxic_both_protected]
#     print("values",len(values))
#     plt.figure(figsize=(10,5))
#     ax = sns.barplot(x=categories, y=values, palette='Blues', edgecolor='w')
#     for i in range(len(values)):
#         ax.text(i, values[i], str(values[i]), ha='center')
# 
#     plt.xticks(rotation=90)
# 
#     plt.ylabel('Count')
#     plt.xlabel('Groups')
#     plt.title('Count of Data by SubGroups')
#     plt.show()
# 
#   def plot_sampeled_data_stat(self):
# 
#     no_toxic_no_protected_unlabel = len(self.data.query('Label == 0 and `gender` == 0 and `race` == 0 and target ==0'))
#     no_toxic_no_protected_label = len(self.data.query('Label == 0 and `gender` == 0 and `race` == 0 and target !=0'))
# 
#     no_toxic_race_protected_unlabel = len(self.data.query('Label == 0 and `gender` == 0 and `race` == 1 and target ==0'))
#     no_toxic_race_protected_label = len(self.data.query('Label == 0 and `gender` == 0 and `race` == 1 and target !=0'))
# 
#     no_toxic_gender_protected_unlabel = len(self.data.query('Label == 0 and `gender` == 1 and `race` == 0 and target ==0'))
#     no_toxic_gender_protected_label = len(self.data.query('Label == 0 and `gender` == 1 and `race` == 0 and target !=0 '))
# 
#     no_toxic_both_protected_unlabel = len(self.data.query('Label == 0 and `gender` == 1 and `race` == 1 and target ==0'))
#     no_toxic_both_protected_label = len(self.data.query('Label == 0 and `gender` == 1 and `race` == 1 and target !=0'))
# 
#     toxic_no_protected_unlabel = len(self.data.query('Label == 1 and `gender` == 0 and `race` == 0 and target ==0'))
#     toxic_no_protected_label = len(self.data.query('Label == 1 and `gender` == 0 and `race` == 0 and target !=0'))
# 
#     toxic_race_protected_unlabel = len(self.data.query('Label == 1 and `gender` == 0 and `race` == 1 and target == 0'))
#     toxic_race_protected_label = len(self.data.query('Label == 1 and `gender` == 0 and `race` == 1 and target !=0 '))
# 
#     toxic_gender_protected_unlabel = len(self.data.query('Label == 1 and `gender` == 1 and `race` == 0 and target ==0'))
#     toxic_gender_protected_label = len(self.data.query('Label == 1 and `gender` == 1 and `race` == 0 and target !=0 '))
# 
#     toxic_both_protected_unlabel = len(self.data.query('Label == 1 and `gender` == 1 and `race` == 1 and target == 0'))
#     toxic_both_protected_label = len(self.data.query('Label == 1 and `gender` == 1 and `race` == 1 and target !=0'))
# 
# 
#     categories = ['No Toxic\ No Protected\ Unlabel','No Toxic\ No Protected\ label', 'No Toxic\ Race Protected\ Unlabel','No Toxic\ Race Protected\ label', 'No Toxic\ Gender Protected\ Unlabel', 'No Toxic\ Gender Protected\ label', 
#                   'No Toxic\ Both Protected\ Unlabel','No Toxic\ Both Protected\ label',
#                   'Toxic\ No Protected\ Unlabel','Toxic\ No Protected\ label', 'Toxic\ Race Protected\ Unlabel','Toxic\ Race Protected\ label', 'Toxic\ Gender Protected\ Unlabel', 'Toxic\ Gender Protected\ label',
#                   'Toxic\ Both Protected\ Unlabel','Toxic\ Both Protected\ label']
#     print("categories",len(categories))
#     values = [ no_toxic_no_protected_unlabel, no_toxic_no_protected_label, no_toxic_race_protected_unlabel, no_toxic_race_protected_label, 
#               no_toxic_gender_protected_unlabel, no_toxic_gender_protected_label,no_toxic_both_protected_unlabel,no_toxic_both_protected_label,
#               toxic_no_protected_unlabel,toxic_no_protected_label,toxic_race_protected_unlabel,toxic_race_protected_label,toxic_gender_protected_unlabel,toxic_gender_protected_label,
#               toxic_both_protected_unlabel,toxic_both_protected_label]
# 
#     short_labels = ['NTNP_U','NTNP_L', 'NTR_U','NTR_L', 'NTG_U', 'NTG_L',
# 'NTBP_U','NTBP_L','TNP_U','TNP_L', 'TR_U','TR_L', 'TG_U', 'TG_L',
# 'TBP_U','TBP_L',]
# 
#     print("values",len(values))
# 
#     plt.figure(figsize=(10,5))
#     fig, axs = plt.subplots(1, 2, figsize=(15, 5))
# 
#     ax = sns.barplot(x=short_labels, y=values, palette='Blues', edgecolor='w', ax=axs[0])
#     for i in range(len(values)):
#         ax.text(i, values[i], str(values[i]), ha='center')
#     ax.set_xticklabels(short_labels, rotation=90)
#     ax.set_ylabel('Count')
#     ax.set_xlabel('Groups')
#     ax.set_title('Count of Data by Groups')
# 
#     mapping_df = pd.DataFrame({'Short Label': short_labels, 'Category': categories})
#     table = axs[1].table(cellText=mapping_df.values, colLabels=mapping_df.columns, cellLoc='center', loc='center')
#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
#     table.scale(1, 1.5)
#     axs[1].axis('tight')
#     axs[1].axis('off')
#     # axs[1].set_title('Legend')
# 
#     plt.show()
# 
#   def plot_generator(self,seed = 1000, dataset = "Noname"):
# 
#      self.single_var_pie_chart(group_col = "Label", group_label_dic ={1: 'Toxic', 0: 'Non-Toxic'}, title = "Label Distribution in subsample from {}".format(dataset),seed = seed, dataset = dataset)
#      self.single_var_pie_chart(group_col = "gender", group_label_dic ={1: 'Gender', 0: 'Non-Gender'}, title = "Gender Subgroup Distribution in subsample from {}".format(dataset),seed = seed, dataset = dataset)
#      self.single_var_pie_chart(group_col = "race", group_label_dic ={1: 'Race', 0: 'Non-Race'}, title = "Race Subgroup Distribution in subsample from {}".format(dataset),seed = seed, dataset = dataset)
# 
#      self.double_var_pie_chart(group_col = ["Label","gender"],  group_label_dic_var1 ={1: 'Toxic', 0: 'Non-Toxic'},
#                            group_label_dic_var2 ={1: 'Gender', 0: 'Non-Gender'}, title = "Label Gender Distribution in subsample from {}".format(dataset),seed = seed, dataset = dataset)
# 
#      self.double_var_pie_chart(group_col = ["Label","race"],  group_label_dic_var1 ={1: 'Toxic', 0: 'Non-Toxic'},
#                            group_label_dic_var2 ={1: 'Race', 0: 'Non-Race'}, title = "Label Race Distribution in subsample from {}".format(dataset),seed = seed, dataset = dataset )
# 
#      self.fine_term(seed = seed, dataset = dataset , title = "Fine Term Distribution in subsample fro {}".format(dataset))
# 
#      self.fine_term_label(seed = seed, dataset = dataset , title = "Fine Term Distribution over Labels in subsample from {}".format(dataset))
#      self.plot_data_stat()
#      self.plot_sampeled_data_stat()
# 
#   def make_autopct(self,values):
#       def my_autopct(pct):
#           total = sum(values)
#           val = int(round(pct*total/100.0))
#           return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
#       return my_autopct
# 
# 
# 
# '''
# 
