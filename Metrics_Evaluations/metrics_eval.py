from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score,balanced_accuracy_score
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from collections import defaultdict
from sklearn.metrics import matthews_corrcoef
from sklearn import metrics

#https://github.com/choprashweta/Adversarial-Debiasing/blob/master/Debiased_Classifier.ipynb
#https://fairlearn.org/main/api_reference/fairlearn.metrics.html

class Metrics:

  def __init__(self, y_pred,actual_labels, protected_labels,thres):
    
    self.y_pred = np.array(y_pred)
    self.actual_labels = np.array(actual_labels)
    self.protected_labels = np.array(protected_labels)
    self.thres = thres
    self.non_protected_labels = Metrics.get_unprotected_labels(protected_labels)


  def get_unprotected_labels(list_of_protected):
    new = [1 if i == 0 else 0 for i in list_of_protected]
    return new


  def get_toxicity_rates(self):
    protected_ops = self.y_pred[self.protected_labels == 1]
    protected_prob = sum(protected_ops)/len(protected_ops)

    non_protected_ops = self.y_pred[self.protected_labels == 0]
    non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)

    return np.round(protected_prob, 2), np.round(non_protected_prob, 2)

  def get_true_positive_rates(self):
    print("self.protected_labels",self.protected_labels)
    print("self.actual_labels",self.actual_labels)
    print("self.y_pred",self.y_pred)


    print("self.protected_labels",sum(self.protected_labels ==1))
    print("self.protected_labels",sum(self.protected_labels ==0))
    print("ssssssss",np.sum(np.bitwise_and(self.protected_labels == 1, self.actual_labels == 1)))

    protected_ops = self.y_pred[np.bitwise_and(self.protected_labels == 1, self.actual_labels == 1)]
    print("protected_ops",protected_ops)
    protected_prob = sum(protected_ops)/len(protected_ops)

    

    non_protected_ops = self.y_pred[np.bitwise_and(self.protected_labels == 0, self.actual_labels == 1)]
    non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)
   
    return np.round(protected_prob, 2), np.round(non_protected_prob, 2)

  def get_false_positive_rates(self):

    protected_ops = self.y_pred[np.bitwise_and(self.protected_labels== 1, self.actual_labels==0)]
    protected_prob = sum(protected_ops)/len(protected_ops)

    non_protected_ops = self.y_pred[np.bitwise_and(self.protected_labels == 0, self.actual_labels == 0)]
    non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)

    return np.round(protected_prob, 2), np.round(non_protected_prob, 2)

  def demographic_parity(self):

    protected_ops = self.y_pred[self.protected_labels== 1]
    protected_prob = sum(protected_ops)/len(protected_ops)

    non_protected_ops = self.y_pred[self.protected_labels == 0]
    non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)

    

    return abs(protected_prob - non_protected_prob)

# | P_protected(y_pred = 1| Y = 1) - P_non-protected(y_pred = 1 | Y = 1) | < self.thres

  def true_positive_parity(self):
    

    protected_ops = self.y_pred[np.bitwise_and(self.protected_labels == 1, self.actual_labels == 1)]
    protected_prob = sum(protected_ops)/len(protected_ops)

    non_protected_ops = self.y_pred [np.bitwise_and(self.protected_labels == 0, self.actual_labels == 1)]
    non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)

    
    return abs(protected_prob - non_protected_prob)


# | P_protected(y_pred = 1| Y = 0) - P_non-protected(y_pred = 1 | Y = 0) | < self.thres

  def false_positive_parity(self):

    protected_ops = self.y_pred[np.bitwise_and(self.protected_labels== 1, self.actual_labels==0)]
    protected_prob = sum(protected_ops)/len(protected_ops)

    non_protected_ops =  self.y_pred [np.bitwise_and(self.protected_labels == 0, self.actual_labels== 0)]
    non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)

    return abs(protected_prob - non_protected_prob)


  # Satisfy both true positive parity and false positive parity
  def equalized_odds(self):

    return self.true_positive_parity() + self.false_positive_parity()

  def equalized_odds_percent(self):

    return 100 - 100*((self.true_positive_parity() + self.false_positive_parity())/2)

  def equalized_opportunity(self):

    return self.true_positive_parity()

  def equalized_opportunity_percent(self):

    return 100 - 100*((self.true_positive_parity())/1)

  
  def p_rule(self):
    y_z_1 =  self.y_pred[self.protected_labels == 1]
    y_z_0 = self.y_pred[self.protected_labels == 0]

    p_1 = sum(y_z_1)/len(y_z_1)
    p_0 = sum(y_z_0)/len(y_z_0)


    if len(y_z_0) == 0 or len(y_z_1) == 0:
      return 0
    else:
      odds = p_1 / p_0
      
      return np.min([odds, 1/odds])


  def err_rate_eq_diff(self,term_coarse,*list_rate):

        err_rate_eq_diff_dic =  defaultdict(int)
        err_rate_eq_diff_total = 0
        err_rate_eq_diff_total += sum(list_rate)
        er_rate_eq_diff = 0
        for idx,rate in enumerate(list_rate):
            err_rate_eq_diff_dic[term_coarse[idx]] = abs(err_rate_eq_diff_total - rate)
            er_rate_eq_diff += abs(err_rate_eq_diff_total - rate)
        
        return er_rate_eq_diff,err_rate_eq_diff_dic
  
  

  def all_metrics(self,coarse_term):

    # try:


      protected_pos_class_rate, non_protected_pos_class_rate = self.get_toxicity_rates()
      protected_tpr, non_protected_tpr = self.get_true_positive_rates()
      protected_fpr, non_protected_fpr = self.get_false_positive_rates()
      demo_parity = self.demographic_parity()
      tp_parity = self.true_positive_parity()
      fp_parity = self.false_positive_parity()
      equ_odds = self.equalized_odds()
      equ_odds_percent = self.equalized_odds_percent()
      equ_opportunity = self.equalized_opportunity()
      equ_opportunity_percent = self.equalized_opportunity_percent()
      p_value = self.p_rule()

      term_coarse = ["protected", "unprotected"]
      fped,fped_dic = self.err_rate_eq_diff(term_coarse,*[protected_tpr,non_protected_tpr])
      fned,fned_dic = self.err_rate_eq_diff(term_coarse,*[protected_fpr,non_protected_fpr])
      

      metrics_dic ={
        coarse_term+ "_p_value": 100*p_value,
        coarse_term+"_demo_parity":demo_parity ,
        coarse_term+"_equ_odds": equ_odds,
        coarse_term+"_equ_odds_percent":equ_odds_percent,
        coarse_term+"_equ_opportunity":equ_opportunity,
        coarse_term+"_equ_opportunity_percent":equ_opportunity_percent,
        # coarse_term+"_protected_pos_class_rate":protected_pos_class_rate,
        # coarse_term+"_non_protected_pos_class_rate": non_protected_pos_class_rate,
        # coarse_term+"_protected_tpr": protected_tpr,
        # coarse_term+"_non_protected_tpr":non_protected_tpr ,
        # coarse_term+"_protected_fpr":protected_fpr ,
        # coarse_term+"_non_protected_fpr":non_protected_fpr ,
        # coarse_term+"_tp_parity":tp_parity ,
        # coarse_term+"_fp_parity":fp_parity ,
        # coarse_term+"_fped": fped,
        # coarse_term+"_fped_dic":fped_dic ,
        # coarse_term+"_fned": fned,
        # coarse_term+"_fned_dic":fned_dic
      }

    # except:

    #    metrics_dic ={
    #     coarse_term+ "_p_value": 0,
    #     coarse_term+"_demo_parity":0 ,
    #     coarse_term+"_equ_odds": 0,
    #     coarse_term+"_equ_odds_percent":0,
    #     coarse_term+"_equ_opportunity":0,
    #     coarse_term+"_equ_opportunity_percent":0}



      return metrics_dic
  
class Metrics_fine_terms:

  def __init__(self, y_pred,actual_labels, protected_labels,fine_terms,fine_terms_list,y_score,thres =  0.5):

    '''
    This class aim to compute the fairness performance for fine-grain terms in the data set. 
    For example, if coarse grain sub group in a dataset is Gender (including samples related to sexual orientation)
    the fine-grain could contains terms such as women, men, LGBTQ+,... . 

    self.y_pred: the predicted labels 
    self.actual_labels: the actual labels
    self.protected_labels: labels for belonging to protected group or not
    self.fine_terms: attribute <list> contain fine-grain information for each sample 
    self.fine_terms_list: general list of of fine terms related to either race or gender for a particualr dataset
    self.thres: threshold


    '''

    self.y_pred = np.array(y_pred)
    self.actual_labels = np.array(actual_labels)
    self.protected_labels = np.array(protected_labels)

    # attr: attribute <list> contain fine-grain information for each sample
    self.fine_terms = fine_terms
    self.y_score = np.array(y_score)

    # general list of of fine terms related to either race or gender for a particualr dataset
    self.fine_terms_list = fine_terms_list

    self.thres = thres

 



  def get_toxicity_rates(self,fine_term):


    protected_ops = self.y_pred[self.fine_terms_np == 1]

    if len(protected_ops) != 0:

        protected_prob = sum(protected_ops)/len(protected_ops)
    else:

      protected_prob = 0

    non_protected_ops = self.y_pred[self.fine_terms_np == 0]

    if len(non_protected_ops) != 0:

        non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)
    else:

        non_protected_prob = 0

    return np.round(protected_prob, 2), np.round(non_protected_prob, 2)

  def get_true_positive_rates(self,fine_term):

    protected_ops = self.y_pred[np.bitwise_and(  self.actual_labels == 1,self.fine_terms_np == 1)]

    if len(protected_ops) != 0:

        protected_prob = sum(protected_ops)/len(protected_ops)
    else:

        protected_prob = 0


    non_protected_ops = self.y_pred[np.bitwise_and(  self.actual_labels == 1,self.fine_terms_np == 0)]

    if len(non_protected_ops) != 0:

        non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)
    else:

        non_protected_prob = 0


    return np.round(protected_prob, 2), np.round(non_protected_prob, 2)


  def get_false_positive_rates(self,fine_term):

    protected_ops = self.y_pred[np.bitwise_and(self.actual_labels==0,self.fine_terms_np == 1)]

    if len(protected_ops) != 0:

        protected_prob = sum(protected_ops)/len(protected_ops)
    else:
        protected_prob = 0

    non_protected_ops = self.y_pred[np.bitwise_and(self.actual_labels == 0,self.fine_terms_np == 0)]

    if len(non_protected_ops) != 0:

        non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)
    else:
        non_protected_prob = 0


    return np.round(protected_prob, 2), np.round(non_protected_prob, 2)

  def demographic_parity(self, fine_term):

    protected_ops = self.y_pred[ self.fine_terms_np == 1]

    if len(protected_ops) != 0:

        protected_prob = sum(protected_ops)/len(protected_ops)
    else:

        protected_prob = 0

    non_protected_ops = self.y_pred[self.fine_terms_np == 0]

    if len(non_protected_ops) != 0:

        non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)
    else:

        non_protected_prob = 0


    return abs(protected_prob - non_protected_prob) #later take absolute diff - but we want to show females predicted more toxic than male

# | P_female(C = 1| Y = 1) - P_male(C = 1 | Y = 1) | < self.thres
  def true_positive_parity(self, fine_term):

    protected_ops = self.y_pred[np.bitwise_and(  self.actual_labels == 1,self.fine_terms_np == 1)]

    if len(protected_ops) != 0:

        protected_prob = sum(protected_ops)/len(protected_ops)
    else:

        protected_prob = 0

    non_protected_ops = self.y_pred [np.bitwise_and(  self.actual_labels == 1,self.fine_terms_np == 0)]

    if len(non_protected_ops) != 0:

        non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)
    else:

        non_protected_prob = 0

    return abs(protected_prob - non_protected_prob) #later take absolute diff - but we want to show females predicted more toxic than male

# | P_female(C = 1| Y = 0) - P_male(C = 1 | Y = 0) | < self.thres

  def false_positive_parity(self, fine_term):

    protected_ops = self.y_pred[np.bitwise_and(  self.actual_labels==0,self.fine_terms_np == 1)]
    # protected_prob = sum(protected_ops)/len(protected_ops)
    if len(protected_ops) != 0:

        protected_prob = sum(protected_ops)/len(protected_ops)
    else:

        protected_prob = 0

    non_protected_ops =  self.y_pred [np.bitwise_and(  self.actual_labels== 0,self.fine_terms_np == 0)]
    # non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)
    if len(non_protected_ops) != 0:

        non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)
    else:

        non_protected_prob = 0

    return abs(protected_prob - non_protected_prob) #later take absolute diff - but we want to show females predicted more toxic than male


  # Satisfy both true positive parity and false positive parity
  def equalized_odds(self,fine_term):

    return self.true_positive_parity(fine_term) + self.false_positive_parity(fine_term)

  
  def equalized_odds_percent(self,fine_term):

    return 100 - 100*((self.true_positive_parity(fine_term) + self.false_positive_parity(fine_term))/2)

  def equalized_opportunity(self,fine_term):

    return self.true_positive_parity(fine_term)

  def equalized_opportunity_percent(self,fine_term):

    return 100- 100*((self.true_positive_parity(fine_term))/1)

  def p_rule(self,fine_term):

    y_z_1 =  self.y_pred[ self.fine_terms_np == 1]
    y_z_0 = self.y_pred[self.fine_terms_np == 0]

    if y_z_0.mean() ==0:
      odds = -1
    elif y_z_1.mean() == 0:
      odds = -1
    else:
      odds = y_z_1.mean() / y_z_0.mean()


    return np.min([odds, 1/odds]) *100


  def err_rate_eq_diff(self,term_coarse,*list_rate):

        err_rate_eq_diff_dic =  defaultdict(int)
        err_rate_eq_diff_total = 0
        err_rate_eq_diff_total += sum(list_rate)
        er_rate_eq_diff = 0
        for idx,rate in enumerate(list_rate):
            err_rate_eq_diff_dic[term_coarse[idx]] = abs(err_rate_eq_diff_total - rate)
            er_rate_eq_diff += abs(err_rate_eq_diff_total - rate)

        return er_rate_eq_diff,err_rate_eq_diff_dic

  


  
  def term_to_np(self,term_list, fine_term):
    '''
    term list: it is list of (list) all the fine terms mentioned in the dataset (consist of all subgroup attributes related to samples.) 
    fine-term: a fine-term that we're looking to find its fairness criteria
    '''
    #define a vector with same size of the inpute data
    fine_terms_np = np.zeros(len(term_list))
    # loop over all the data, if it contains the fine term, then assign it to one
    # <change> term_list if it's a list of attributre, we need to look if all the samples in
    # that list is eq to fine_term
    for idx,terms in enumerate(term_list):
        if fine_term in terms :
          # print("inside",terms)
          # print("inside",fine_term)
          fine_terms_np[idx] = 1
    return fine_terms_np



  def all_metrics(self,term_fine):

    # a vector that shows either a sample contains terms_fine [ex: women] or not
    # term_fine is a  particular term in a find-term_list
    # self.fine_terms is list of list of fine terms for all the samples
    self.fine_terms_np = self.term_to_np(self.fine_terms, term_fine)
    # self.fine_terms_np contains information about the frequency of the fine-terms in the dataset (validation set) 
    # print("self.fine_terms_np",self.fine_terms_np.shape)
    # print("self.y_pred",self.y_pred)
    # print("self.y_pred",list(self.fine_terms_np == 1))
    # print(" self.y_pred[self.fine_terms_np == 1]", self.y_pred[list(self.fine_terms_np) == 1])


    '''
    add eq funvtion here and to the dic
    see how accuracy metrics endded up here
    after you finish with this you must go make things write the main code
    '''
    protected_pos_class_rate, non_protected_pos_class_rate = self.get_toxicity_rates(term_fine)
    protected_tpr, non_protected_tpr = self.get_true_positive_rates(term_fine)
    protected_fpr, non_protected_fpr = self.get_false_positive_rates(term_fine)
    demo_parity = self.demographic_parity(term_fine)
    tp_parity = self.true_positive_parity(term_fine)
    fp_parity = self.false_positive_parity(term_fine)
    equ_odds = self.equalized_odds(term_fine)
    equ_odds_percent = self.equalized_odds_percent(term_fine)
    equ_opportunity = self.equalized_opportunity(term_fine)
    equ_opportunity_percent = self.equalized_opportunity_percent(term_fine)
    p_value = self.p_rule(term_fine)
    term_coarse = ["protected", "unprotected"]
    fped,fped_dic = self.err_rate_eq_diff(term_coarse,*[protected_tpr,non_protected_tpr])
    fned,fned_dic = self.err_rate_eq_diff(term_coarse,*[protected_fpr,non_protected_fpr])

    print("number of samples for {}".format(term_fine),sum(self.fine_terms_np == 1))

    y_pred = self.y_pred[ self.fine_terms_np == 1]
    actual_labels = self.actual_labels[self.fine_terms_np == 1]
    y_score = self.y_score[self.fine_terms_np == 1]


    acc_performance = Performance_Metrics.metric_cal(actual_labels, y_pred)

    # balance_acc, acc,pre,rec,f1 = Performance_Metrics.get_metrics(actual_labels, y_pred)

    # mc = matthews_corrcoef(actual_labels, y_pred)

    # acc_performance = {"accuracy":acc,"Balanced Accuracy":balance_acc,"Precision":pre,"Recall":rec,"F1":f1,"MC":mc}
    # print("term_fine",term_fine)
    # print("actual_labels",actual_labels)
    # print("y_score",y_score)

    if sum(self.fine_terms_np == 1) != 0:


        fpr, tpr, _ = metrics.roc_curve(actual_labels, y_score, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        confusion_matrix = metrics.confusion_matrix(actual_labels, y_pred)

        # acc_performance["roc_auc"] = roc_auc
        # acc_performance["confusion_matrix"] = confusion_matrix

        metrics_dic ={
            "p_value": p_value,
            "demo_parity":demo_parity ,
            "equ_odds": equ_odds,
            "equ_odds_percent": equ_odds_percent,
            "equ_opportunity": equ_opportunity,
            "equ_opportunity_percent": equ_opportunity_percent,
            "number of samples":sum(self.fine_terms_np == 1),
            # "protected_pos_class_rate":protected_pos_class_rate,
            # "non_protected_pos_class_rate": non_protected_pos_class_rate,
            # "protected_tpr": protected_tpr,
            # "non_protected_tpr":non_protected_tpr ,
            # "protected_fpr":protected_fpr ,
            # "non_protected_fpr":non_protected_fpr ,
            # "tp_parity":tp_parity ,
            # "fp_parity":fp_parity ,
            # "fped": fped,
            # "fped_dic":fped_dic ,
            # "fned": fned,
            # "fned_dic":fned_dic
        }

    else:

       metrics_dic ={
            "p_value": 0,
            "demo_parity":0 ,
            "equ_odds": -1,
            "equ_odds_percent": 0,
            "equ_opportunity": -1,
            "equ_opportunity_percent": 0,
            "number of samples":sum(self.fine_terms_np == 1)}


        

    metrics_dic = {**metrics_dic, **acc_performance}




    return metrics_dic

  def all_metrics_terms(self):
    fine_metrics = {}

    #for each term in the general fine terms list
    for fine_term in self.fine_terms_list:

      #compute the fairness matric for that particular term
      fine_metrics[fine_term] = self.all_metrics(fine_term)
      # <change> maybe you can add accuracy performance terms here as well

    return fine_metrics

  
  
class Performance_Metrics:

  # @staticmethod
  # def get_metrics(labels, preds):
  #   # pred_flat = preds.flatten()
  #   # labels_flat = labels.flatten()
  #   pred_flat = preds
  #   labels_flat = labels

  #   balance_acc = balanced_accuracy_score(labels_flat, pred_flat)
  #   acc = accuracy_score(labels_flat, pred_flat)
  #   pre = precision_score(labels_flat, pred_flat)
  #   rec = recall_score(labels_flat, pred_flat)
  #   f1 = f1_score(labels_flat, pred_flat, average="weighted")
  #   mc = matthews_corrcoef(labels_flat, pred_flat)

  #   acc_performance = {"Accuracy":acc,"Balanced Accuracy":balance_acc,"Precision":pre,"Recall":rec,"F1":f1,"MC":mc}

  #   return acc_performance

  @staticmethod
  def get_metrics(labels, preds, gender_label, race_label):
    # pred_flat = preds.flatten()
    # labels_flat = labels.flatten()
    labels = np.array(labels)
    preds = np.array(preds)
    gender_label = np.array(gender_label)
    race_label = np.array(race_label)

    pred_flat_gender = preds[gender_label == 1]
    labels_flat_gender = labels[gender_label == 1]

    gender = Performance_Metrics.metric_cal(labels_flat_gender, pred_flat_gender, term = 'gender_')

    pred_flat_non_gender = preds[gender_label == 0]
    labels_flat_non_gender = labels[gender_label == 0]
    non_gender = Performance_Metrics.metric_cal(labels_flat_non_gender, pred_flat_non_gender, term = 'non_gender_')


    pred_flat_race = preds[race_label == 1]
    labels_flat_race = labels[race_label == 1]
    race = Performance_Metrics.metric_cal(labels_flat_race, pred_flat_race, term = 'race_')


    pred_flat_non_race = preds[race_label == 0]
    labels_flat_non_race = labels[race_label == 0]
    non_race = Performance_Metrics.metric_cal(labels_flat_non_race, pred_flat_non_race, term = 'non_race_')



    pred_flat_sensitive = preds[np.bitwise_and(  gender_label == 1,gender_label == 1)]
    labels_flat_sensitive = labels[np.bitwise_and(  gender_label == 1,gender_label == 1)]
    sensitive = Performance_Metrics.metric_cal(labels_flat_sensitive, pred_flat_sensitive, term = 'sensitive_')


    pred_flat_non_sensitive = preds[np.bitwise_and(  gender_label == 0,gender_label == 0)]
    labels_flat_non_sensitive = labels[np.bitwise_and(  gender_label == 0,gender_label == 0)]
    non_sensitive  = Performance_Metrics.metric_cal(labels_flat_non_sensitive, pred_flat_non_sensitive, term = 'non_sensitive_')

    total = Performance_Metrics.metric_cal(labels,preds)

    acc_performance = {**total,**gender,**non_gender,**race,**non_race,**sensitive,**non_sensitive}

    return acc_performance
  
  def metric_cal(labels_flat, pred_flat, term = ''):
    if len(term) !=0: 
    
      balance_acc = balanced_accuracy_score(labels_flat, pred_flat)
      acc = accuracy_score(labels_flat, pred_flat)
      pre = precision_score(labels_flat, pred_flat)
      rec = recall_score(labels_flat, pred_flat)
      f1 = f1_score(labels_flat, pred_flat, average="weighted")
      mc = matthews_corrcoef(labels_flat, pred_flat)

      acc_performance = {term+"Accuracy":100*acc,term+"Balanced Accuracy":100*balance_acc,term+"Precision":100*pre,term+"Recall":100*rec,term+"F1":100*f1,term+"MC":mc}

    else:
     

        balance_acc = balanced_accuracy_score(labels_flat, pred_flat)
        acc = accuracy_score(labels_flat, pred_flat)
        pre = precision_score(labels_flat, pred_flat)
        rec = recall_score(labels_flat, pred_flat)
        f1 = f1_score(labels_flat, pred_flat, average="weighted")
        mc = matthews_corrcoef(labels_flat, pred_flat)

        acc_performance = {"Accuracy":100*acc,"Balanced Accuracy":100*balance_acc,"Precision":100*pre,"Recall":100*rec,"F1":100*f1,"MC":mc}



    return acc_performance 




# from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score,balanced_accuracy_score
# import pandas as pd
# import numpy as np
# import seaborn as sns 
# import matplotlib.pyplot as plt 
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
# from collections import defaultdict

# #https://fairlearn.org/main/api_reference/fairlearn.metrics.html

# class Metrics:

#   def __init__(self, y_pred,actual_labels, protected_labels,thres):
    
#     self.y_pred = y_pred
#     self.actual_labels = actual_labels
#     self.protected_labels = protected_labels
#     self.thres = thres
#     self.non_protected_labels = Metrics.get_unprotected_labels(protected_labels)


#   def get_unprotected_labels(list_of_protected):
#     new = [1 if i == 0 else 0 for i in list_of_protected]
#     return new


#   def get_toxicity_rates(self):
#     protected_ops = self.y_pred[self.protected_labels == 1]
#     # print("Number of protected attributes with predicted label(    self.y_pred)=1 in test set",sum(protected_ops))
#     # print("Number of protected attributes in test set",len(protected_ops))

  

#     protected_prob = sum(protected_ops)/len(protected_ops)

#     non_protected_ops = self.y_pred[self.protected_labels == 0]
 
#     non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)

#     return np.round(protected_prob, 2), np.round(non_protected_prob, 2)

#   def get_true_positive_rates(self):

#     protected_ops = self.y_pred[np.bitwise_and(self.protected_labels == 1, self.actual_labels == 1)]
#     protected_prob = sum(protected_ops)/len(protected_ops)

#     # print("Number of protected attributes with predicted label(    self.y_pred)=1 and actaul label = 1 in test set",protected_ops)
#     # print("Number of protected attributes with actaul label = 1 in test set",len(protected_ops))


#     non_protected_ops = self.y_pred[np.bitwise_and(self.protected_labels == 0, self.actual_labels == 1)]
#     non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)
#     # print("sum(non_protected_ops)",sum(non_protected_ops))
#     # print("len(non_protected_ops)",len(non_protected_ops))


#     return np.round(protected_prob, 2), np.round(non_protected_prob, 2)

# # wronggggggggg
#   def get_false_positive_rates(self):

#     protected_ops = self.y_pred[np.bitwise_and(self.protected_labels== 1, self.actual_labels==0)]
#     protected_prob = sum(protected_ops)/len(protected_ops)

#     non_protected_ops = self.y_pred[np.bitwise_and(self.protected_labels == 0, self.actual_labels == 0)]
#     non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)

#     return np.round(protected_prob, 2), np.round(non_protected_prob, 2)

#   def demographic_parity(self):

#     protected_ops = self.y_pred[self.protected_labels== 1]
#     protected_prob = sum(protected_ops)/len(protected_ops)

#     non_protected_ops = self.y_pred[self.protected_labels == 0]
#     non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)

#     # print("demographic_parity, probability of P_protected (self.y_pred= 1) over all predictions ",protected_prob)
#     # print("demographic_parity, probability of P_unprotected (self.y_pred = 0) over all predictions ",non_protected_prob)


#     return abs(protected_prob - non_protected_prob) #later take absolute diff - but we want to show females predicted more toxic than male

# # | P_female(C = 1| Y = 1) - P_male(C = 1 | Y = 1) | < self.thres

#   def true_positive_parity(self):

#     protected_ops = self.y_pred[np.bitwise_and(self.protected_labels == 1, self.actual_labels == 1)]
#     protected_prob = sum(protected_ops)/len(protected_ops)

#     non_protected_ops = self.y_pred [np.bitwise_and(self.protected_labels == 0, self.actual_labels == 1)]
#     non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)

#     # print("true_positive_parity,  P_female(    self.y_pred= 1| Y = 1) ",protected_prob)
#     # print("true_positive_parity, P_male(    self.y_pred= 1 | Y = 1) ",non_protected_prob)


#     return abs(protected_prob - non_protected_prob) #later take absolute diff - but we want to show females predicted more toxic than male

# # | P_female(C = 1| Y = 0) - P_male(C = 1 | Y = 0) | < self.thres

# #maybe wrong

#   def false_positive_parity(self):

#     protected_ops = self.y_pred[np.bitwise_and(self.protected_labels== 1, self.actual_labels==0)]
#     protected_prob = sum(protected_ops)/len(protected_ops)

#     non_protected_ops =  self.y_pred [np.bitwise_and(self.protected_labels == 0, self.actual_labels== 0)]
#     non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)

#     return abs(protected_prob - non_protected_prob) #later take absolute diff - but we want to show females predicted more toxic than male

  
#   # Satisfy both true positive parity and false positive parity
#   def equalized_odds(self):

#     return self.true_positive_parity() + self.false_positive_parity()

#   ##############################
#   def p_rule(self):
#     y_z_1 =  self.y_pred[self.protected_labels == 1]
#     y_z_0 = self.y_pred[self.protected_labels == 0]

#     # print("y_z_1", len(y_z_1),"y_z_0",len(y_z_0))
#     # print("total y_z_1: ", len(self.y_pred[self.y_pred == 1]))
#     # print("total y_z_0: ",len(self.y_pred[self.y_pred == 0]) )

#     p_1 = sum(y_z_1)/len(y_z_1)
#     p_0 = sum(y_z_0)/len(y_z_0)


#     if len(y_z_0) == 0 or len(y_z_1) == 0:
#       return 0
#     else:
#       odds = p_1 / p_0
      
#       return np.min([odds, 1/odds])


#   def err_rate_eq_diff(self,term_coarse,*list_rate):
#         err_rate_eq_diff_dic =  defaultdict(int)
#         err_rate_eq_diff_total = 0
#         err_rate_eq_diff_total += sum(list_rate)
#         er_rate_eq_diff = 0
#         for idx,rate in enumerate(list_rate):
#             err_rate_eq_diff_dic[term_coarse[idx]] = abs(err_rate_eq_diff_total - rate)
#             er_rate_eq_diff += abs(err_rate_eq_diff_total - rate)
        
#         return er_rate_eq_diff,err_rate_eq_diff_dic
  
  

#   def all_metrics(self):
#     protected_pos_class_rate, non_protected_pos_class_rate = self.get_toxicity_rates()
#     protected_tpr, non_protected_tpr = self.get_true_positive_rates()
#     protected_fpr, non_protected_fpr = self.get_false_positive_rates()
#     demo_parity = self.demographic_parity()
#     tp_parity = self.true_positive_parity()
#     fp_parity = self.false_positive_parity()
#     equ_odds = self.equalized_odds()
#     p_value = self.p_rule()
#     term_coarse = ["protected", "unprotected"]
#     fped,fped_dic = self.err_rate_eq_diff(term_coarse,*[protected_tpr,non_protected_tpr])
#     fned,fned_dic = self.err_rate_eq_diff(term_coarse,*[protected_fpr,non_protected_fpr])
    

#     metrics_dic ={
#         "p_value": p_value,
#         "protected_pos_class_rate":protected_pos_class_rate,
#         "non_protected_pos_class_rate": non_protected_pos_class_rate,
#         "protected_tpr": protected_tpr,
#         "non_protected_tpr":non_protected_tpr ,
#         "protected_fpr":protected_fpr ,
#         "non_protected_fpr":non_protected_fpr ,
#         "demo_parity":demo_parity ,
#         "tp_parity":tp_parity ,
#         "fp_parity":fp_parity ,
#         "equ_odds": equ_odds,
#         "fped": fped,
#         "fped_dic":fped_dic ,
#         "fned": fned,
#         "fned_dic":fned_dic
#     }

#     return metrics_dic
  
# class Metrics_fine_terms:

#   def __init__(self, y_pred,actual_labels, protected_labels,fine_terms,fine_terms_list,thres):
    
#     self.y_pred = y_pred
#     self.actual_labels = actual_labels
#     self.protected_labels = protected_labels
#     self.fine_terms_list = fine_terms_list
#     self.fine_terms = fine_terms
#     self.thres = thres

#   def term_to_np(term_list, fine_term):
#     fine_terms_np = np.zeros(len(term_list))
#     for idx in range(len(term_list)):
#       if term_list[idx] == fine_term:
#         fine_terms_np[idx] = 1
#     return fine_terms_np



#   def get_toxicity_rates(self,fine_term):
        
#     protected_ops = self.y_pred[np.bitwise_and(self.protected_labels == 1,self.fine_terms_np == 1)]  

#     if len(protected_ops) != 0:

#         protected_prob = sum(protected_ops)/len(protected_ops)
#     else:

#       protected_prob = 0

#     non_protected_ops = self.y_pred[np.bitwise_and(self.protected_labels == 0,self.fine_terms_np == 0)]

#     if len(non_protected_ops) != 0:

#         non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)
#     else:

#         non_protected_prob = 0

#     return np.round(protected_prob, 2), np.round(non_protected_prob, 2)

#   def get_true_positive_rates(self,fine_term):

#     protected_ops = self.y_pred[np.bitwise_and(self.protected_labels == 1, self.actual_labels == 1,self.fine_terms_np == 1)]
#     # protected_prob = sum(protected_ops)/len(protected_ops)

#     if len(protected_ops) != 0:

#         protected_prob = sum(protected_ops)/len(protected_ops)
#     else:

#         protected_prob = 0


#     non_protected_ops = self.y_pred[np.bitwise_and(self.protected_labels == 0, self.actual_labels == 1,self.fine_terms_np == 0)]
#     # non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)


#     if len(non_protected_ops) != 0:

#         non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)
#     else:

#         non_protected_prob = 0


#     return np.round(protected_prob, 2), np.round(non_protected_prob, 2)


#   def get_false_positive_rates(self,fine_term):

#     protected_ops = self.y_pred[np.bitwise_and(self.protected_labels== 1, self.actual_labels==0,self.fine_terms_np == 1)]
#     # protected_prob = sum(protected_ops)/len(protected_ops)
#     if len(protected_ops) != 0:

#         protected_prob = sum(protected_ops)/len(protected_ops)
#     else:

#         protected_prob = 0

#     non_protected_ops = self.y_pred[np.bitwise_and(self.protected_labels == 0, self.actual_labels == 0,self.fine_terms_np == 0)]
#     # non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)

#     if len(non_protected_ops) != 0:

#         non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)
#     else:

#         non_protected_prob = 0


#     return np.round(protected_prob, 2), np.round(non_protected_prob, 2)

#   #compare it with toxicity rate, one might change
#   def demographic_parity(self, fine_term):

#     protected_ops = self.y_pred[np.bitwise_and(self.protected_labels== 1,self.fine_terms_np == 1)]
#     # protected_prob = sum(protected_ops)/len(protected_ops)

#     if len(protected_ops) != 0:

#         protected_prob = sum(protected_ops)/len(protected_ops)
#     else:

#         protected_prob = 0

#     non_protected_ops = self.y_pred[np.bitwise_and(self.protected_labels== 0,self.fine_terms_np == 0)]
#     # non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)
#     if len(non_protected_ops) != 0:

#         non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)
#     else:

#         non_protected_prob = 0


#     return abs(protected_prob - non_protected_prob) #later take absolute diff - but we want to show females predicted more toxic than male

# # | P_female(C = 1| Y = 1) - P_male(C = 1 | Y = 1) | < self.thres

#   def true_positive_parity(self, fine_term):

#     protected_ops = self.y_pred[np.bitwise_and(self.protected_labels == 1, self.actual_labels == 1,self.fine_terms_np == 1)]
#     # protected_prob = sum(protected_ops)/len(protected_ops)

#     if len(protected_ops) != 0:

#         protected_prob = sum(protected_ops)/len(protected_ops)
#     else:

#         protected_prob = 0

#     non_protected_ops = self.y_pred [np.bitwise_and(self.protected_labels == 0, self.actual_labels == 1,self.fine_terms_np == 0)]
#     # non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)
#     if len(non_protected_ops) != 0:

#         non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)
#     else:

#         non_protected_prob = 0

#     return abs(protected_prob - non_protected_prob) #later take absolute diff - but we want to show females predicted more toxic than male

# # | P_female(C = 1| Y = 0) - P_male(C = 1 | Y = 0) | < self.thres

#   def false_positive_parity(self, fine_term):

#     protected_ops = self.y_pred[np.bitwise_and(self.protected_labels== 1, self.actual_labels==0,self.fine_terms_np == 1)]
#     # protected_prob = sum(protected_ops)/len(protected_ops)
#     if len(protected_ops) != 0:

#         protected_prob = sum(protected_ops)/len(protected_ops)
#     else:

#         protected_prob = 0

#     non_protected_ops =  self.y_pred [np.bitwise_and(self.protected_labels == 0, self.actual_labels== 0,self.fine_terms_np == 0)]
#     # non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)
#     if len(non_protected_ops) != 0:

#         non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)
#     else:

#         non_protected_prob = 0

#     return abs(protected_prob - non_protected_prob) #later take absolute diff - but we want to show females predicted more toxic than male

  
#   # Satisfy both true positive parity and false positive parity
#   def equalized_odds(self,fine_term):

#     return self.true_positive_parity(fine_term) + self.false_positive_parity(fine_term)

#   ##############################
#   def p_rule(self,fine_term):
#     y_z_1 =  self.y_pred[np.bitwise_and(self.protected_labels == 1,self.fine_terms_np == 1)]
#     y_z_0 = self.y_pred[np.bitwise_and(self.protected_labels == 0,self.fine_terms_np == 0)]
#     if y_z_0.mean() ==0:
#       odds = -1
#     elif y_z_1.mean() == 0:
#       odds = -1
#     else:
#       odds = y_z_1.mean() / y_z_0.mean()


#     return np.min([odds, 1/odds]) *100


#   def err_rate_eq_diff(self,term_coarse,*list_rate):
#         err_rate_eq_diff_dic =  defaultdict(int)
#         err_rate_eq_diff_total = 0
#         err_rate_eq_diff_total += sum(list_rate)
#         er_rate_eq_diff = 0
#         for idx,rate in enumerate(list_rate):
#             err_rate_eq_diff_dic[term_coarse[idx]] = abs(err_rate_eq_diff_total - rate)
#             er_rate_eq_diff += abs(err_rate_eq_diff_total - rate)
        
#         return er_rate_eq_diff,err_rate_eq_diff_dic
  
  

#   def all_metrics(self,term_fine):

#     self.fine_terms_np = Metrics_fine_terms.term_to_np(self.fine_terms, term_fine)
#     # print(self.fine_terms_np == 1)
#     protected_pos_class_rate, non_protected_pos_class_rate = self.get_toxicity_rates(term_fine)
#     protected_tpr, non_protected_tpr = self.get_true_positive_rates(term_fine)
#     protected_fpr, non_protected_fpr = self.get_false_positive_rates(term_fine)
#     demo_parity = self.demographic_parity(term_fine)
#     tp_parity = self.true_positive_parity(term_fine)
#     fp_parity = self.false_positive_parity(term_fine)
#     equ_odds = self.equalized_odds(term_fine)
#     p_value = self.p_rule(term_fine)
#     term_coarse = ["protected", "unprotected"]
#     fped,fped_dic = self.err_rate_eq_diff(term_coarse,*[protected_tpr,non_protected_tpr])
#     fned,fned_dic = self.err_rate_eq_diff(term_coarse,*[protected_fpr,non_protected_fpr])
    

#     metrics_dic ={
#         "p_value": p_value,
#         "protected_pos_class_rate":protected_pos_class_rate,
#         "non_protected_pos_class_rate": non_protected_pos_class_rate,
#         "protected_tpr": protected_tpr,
#         "non_protected_tpr":non_protected_tpr ,
#         "protected_fpr":protected_fpr ,
#         "non_protected_fpr":non_protected_fpr ,
#         "demo_parity":demo_parity ,
#         "tp_parity":tp_parity ,
#         "fp_parity":fp_parity ,
#         "equ_odds": equ_odds,
#         "fped": fped,
#         "fped_dic":fped_dic ,
#         "fned": fned,
#         "fned_dic":fned_dic
#     }



#     return metrics_dic
  
#   def all_metrics_terms(self):
#     fine_metrics = {}
#     for fine_term in self.fine_terms_list:

#       # print("loop",fine_term)
#       fine_metrics[fine_term] = self.all_metrics(fine_term)

#     return fine_metrics


  
  
# class Performance_Metrics:

#   def get_metrics(labels, preds):
#     pred_flat = preds.flatten()
#     labels_flat = labels.flatten()

#     balance_acc = balanced_accuracy_score(labels_flat, pred_flat)
#     acc = accuracy_score(labels_flat, pred_flat)
#     pre = precision_score(labels_flat, pred_flat)
#     rec = recall_score(labels_flat, pred_flat)
#     f1 = f1_score(labels_flat, pred_flat, average="weighted")

#     # metrics_dic ={
#     #     "balance_acc": balance_acc,
#     #     "acc":acc,
#     #     "pre": pre,
#     #     "rec": rec,
#     #     "f1":f1 
#     # }
#     # print(metrics_dic)

#     return balance_acc, acc,pre,rec,f1



