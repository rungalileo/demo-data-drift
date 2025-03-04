import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def false_pos_rate(trues, preds):
  if np.sum(trues == 0) == 0:
    return None
  return np.sum((trues == 0) & (preds == 1)) / np.sum(trues == 0)

def false_neg_rate(trues, preds):
  if np.sum(trues == 1) == 0:
    return None
  return np.sum((trues == 1) & (preds == 0)) / np.sum(trues == 1)

def recall(trues, preds):
  if np.sum(trues == 1) == 0:
    return None
  return np.sum((trues == 1) & (preds == 1)) / np.sum(trues == 1)

def evaluate(trues, preds, labels=None):
  trues = np.array(trues)
  preds = np.array(preds)
  if labels:
    labels = np.array(labels)

  metrics = {}
  metrics['n'] = trues.shape[0]
  metrics['pc_drift'] = np.sum(trues)/trues.shape[0]
  metrics['precision'] = precision_score(trues, preds)
  metrics['recall'] = recall_score(trues, preds)
  metrics['f1'] = f1_score(trues, preds)

  if labels is not None:
    for data_type in np.unique(labels):
      trues_ = trues[labels==data_type]
      preds_ = preds[labels==data_type]

      metrics[f"{data_type}_fpr"] = false_pos_rate(trues_, preds_)
      metrics[f"{data_type}_fnr"] = false_neg_rate(trues_, preds_)
      metrics[f"{data_type}_recall"] = recall(trues_, preds_)


  return metrics