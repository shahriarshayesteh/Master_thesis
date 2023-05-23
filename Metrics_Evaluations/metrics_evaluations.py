from sklearn.metrics import accuracy_score
from collections import defaultdict
import sys
sys.path.append('/content/drive/MyDrive/Master_thesis/Metrics_Evaluations') 
import model_bias_analysis
import re
import numpy as np
#make a class inside metrics for util methods or other design patterns

class Evaluation:

  def __init__(self,data_set, terms, model,tokenizer, seq_max_len = 250 ):

    self.data_set = data_set
    self.terms = terms
    self.model = model
    self.tokenizer = tokenizer
    self.seq_max_len = seq_max_len
    # need to change the function to eliminate text in the function representation instead use IPPTS[text]
    model_bias_analysis.add_subgroup_columns_from_text(self.data_set, 'text', self.terms)
    self.pred_test_set()
    self.group_matrix_build()


  def pred_test_set(self):

    test_IPPTS = self.tokenizer.text_to_seq(self.data_set['text'],self.seq_max_len)

    self.labels_IPPTS= self.data_set["label"].values.reshape(-1, 1) * 1.0

    pred = self.model.predict(test_IPPTS)

    self.y_pred =self.pred_label(pred)



  def group_matrix_build(self):
    names = [re.compile(" %s " % term) for term in self.terms]

    idxs = []
    self.group_matrix =np.zeros((self.data_set['text'].shape[0],len(names)))
    for i in range(len(self.data_set['text'])):
        for j in range(len(names)):
            self.group_matrix[i, j] = len(names[j].findall(self.data_set['text'][i]))

  
  def pred_label(self,predictions):
    y_pred = np.zeros((predictions.shape))
    for i,pred in enumerate(predictions):
      if pred < 0.5:
        y_pred[i] = 0
      else:
        y_pred[i] = 1

    return y_pred 

  def term_rates_dic(self,eval_rate_list):
    eval_dic = defaultdict(int)

    for idx,term in enumerate(self.terms):

      eval_dic[term] = eval_rate_list[idx]

    return eval_dic
  
  


  def evaluation(self):

    fpr_terms = Metric.group_false_positive_rates(self.labels_IPPTS,self.y_pred,self.group_matrix)
    fnr_terms = Metric.group_false_negative_rates(self.labels_IPPTS,self.y_pred,self.group_matrix)


    eval_fpr_dic = self.term_rates_dic(fpr_terms)
    eval_fnr_dic = self.term_rates_dic(fnr_terms)

    fpr_eec_dict, fnr_eec_dict, auc_eec_dict = defaultdict(int), defaultdict(int), defaultdict(int)



    fnr_eec = Metric.false_negative_rate(self.labels_IPPTS,self.y_pred)
    efnr, eval_fnr_dic_all = Metric.err_rate_eq_diff(fnr_eec,eval_fnr_dic ,self.terms)


    fpr_eec = Metric.false_positive_rate(self.labels_IPPTS,self.y_pred)
    efpr, eval_fpr_dic_all = Metric.err_rate_eq_diff(fpr_eec,eval_fpr_dic ,self.terms)

    return eval_fnr_dic_all,eval_fpr_dic_all





class Metric:

  def __init__(self):
    pass

  def acc_score(labels, predictions):
      return accuracy_score(labels,predictions)


  def error_rate(labels, predictions):
    # Returns error rate for given labels and predictions.
    # Recall that the labels are binary (0 or 1).
    signed_labels = (labels * 2) - 1
    return np.mean(signed_labels * predictions <= 0.0)


  def false_negative_rate(labels, predictions):
      # Returns false negative rate for given labels and predictions.
      if np.sum(labels > 0) == 0:  # Any positives?
          return 0.0
      else:
          return np.mean(predictions[labels > 0] <= 0)


  def false_positive_rate(labels, predictions):
      # Returns false positive rate for given labels and predictions.
      if np.sum(labels <= 0) == 0:  # Any negatives?
          return 0.0
      else:
          return np.mean(predictions[labels <= 0] > 0)


  def group_false_negative_rates(labels, predictions, groups):
      # Returns list of per-group false negative rates for given labels,
      # predictions and group membership matrix.
      fnrs = []
      for ii in range(groups.shape[1]):
          labels_ii = labels[groups[:, ii] == 1]
          if np.sum(labels_ii > 0) > 0:  # Any positives?
              predictions_ii = predictions[groups[:, ii] == 1]
              fnr_ii = np.mean(predictions_ii[labels_ii > 0] <= 0)
          else:
              fnr_ii = 0.0
          fnrs.append(fnr_ii)
      return fnrs


  def group_false_positive_rates(labels, predictions, groups):
      # Returns list of per-group false positive rates for given labels,
      # predictions and group membership matrix.
      fprs = []
      for ii in range(groups.shape[1]):
          labels_ii = labels[groups[:, ii] == 1]
          if np.sum(labels_ii <= 0) > 0:  # Any negatives?
              predictions_ii = predictions[groups[:, ii] == 1]
              fpr_ii = np.mean(predictions_ii[labels_ii <= 0] > 0)
          else:
              fpr_ii = 0.0
          fprs.append(fpr_ii)
      return fprs

  def err_rate_eq_diff(false_total_rate,false_terms_rate,terms):
      err_rate_eq_diff_dic =  defaultdict(int)
      err_rate_eq_diff_dic["all"] += false_total_rate
      er_rate_eq_diff = 0
      for idx,term in enumerate(terms):
          err_rate_eq_diff_dic[term] += false_terms_rate[term]
          er_rate_eq_diff += abs(false_total_rate - false_terms_rate[term])
      
      return er_rate_eq_diff,err_rate_eq_diff_dic
