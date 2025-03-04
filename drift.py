from typing import List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.contrib import itertools
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

from metrics import evaluate

def _to_list(arr):
  if isinstance(arr, np.ndarray):
    return arr.tolist()
  elif isinstance(arr, list):
    return arr
  else:
    raise Warning(f"Unsupported type {type(arr)}")
    return arr

def value_to_probability(x, threshold=0.5, k=10):
  """Maps x in [0,2] to probability given input threshold.
  By definition, probability at x=threhsold = 0.5

  Adjust k accordingly:
    - for higher values of k: probability goes to 0/1 faster as we move away from the threhold
  """
  return 1 / (1 + np.exp(-k * (x - threshold)))

class NNDriftEstimator:
  def __init__(self, embedder):
    self.embedder = embedder

    self.query_cache = [] # store new queries
    self.embedding_cache = []

  def fit(self, reference_data, k, pc):
    self.reference_data = reference_data
    self.k = k
    self.pc = pc

    self.reference_embeddings = self.embedder.encode(reference_data)
    self.knn = NearestNeighbors(n_neighbors=self.k, metric='cosine').fit(self.reference_embeddings)

    distances, indices = self.knn.kneighbors(self.reference_embeddings)
    self.threshold = pd.Series(distances[:,-1]).quantile(self.pc)

    # dists = distances[:,-1]
    # threshold = np.mean(distances[:,-1]) + 2.5 * np.std(distances[:,-1])

    return self

  def tune_parameters(self, reference_data, ood_source):
    self.reference_data = reference_data
    # grid search over K and PERCENTILE parameters

    # sample IND samples
    # Same IND, OOD data
    train_data, test_ind_data = train_test_split(reference_data, test_size=0.1)
    test_ood_data = np.random.choice(ood_source, size=len(test_ind_data), replace=False)
    
    # Generate Synthetic IND, OOD Data
    # train_data = reference_data
    # n_test = int(0.2*len(train_data))
    # test_ind_data = np.random.choice(verizon_rephrases, size=n_test, replace=False)
    # test_ood_data = np.random.choice(ood_source, size=n_test, replace=False)

    test_data = np.concatenate([test_ind_data, test_ood_data])
    test_labels = [0]*len(test_ind_data) + [1]*len(test_ood_data)
    test_types = ["IND"]*len(test_ind_data) + ["OOD"]*len(test_ood_data)

    reference_embeddings = self.embedder.encode(train_data)
    test_embeddings = self.embedder.encode(test_data)

    k_range = np.linspace(1, 0.1*len(reference_embeddings), 10).astype(int)# up to 10% of dataset
    pc_range = [0.8, 0.85, 0.9, 0.95]

    tuning_table = []
    best_f1 = 0
    best_params = []
    for k, pc in itertools.product(k_range, pc_range):

      knn = NearestNeighbors(n_neighbors=k, metric='cosine').fit(reference_embeddings)
      # set threshold
      distances, indices = knn.kneighbors(reference_embeddings)
      threshold = pd.Series(distances[:,-1]).quantile(pc)

      # evaluate
      distances, indices = knn.kneighbors(test_embeddings)  
      probabilities = value_to_probability(distances[:,-1], threshold=threshold)
      predictions = distances[:,-1] > threshold
      metrics = evaluate(test_labels, predictions, test_types)

      metrics['roc_auc_score'] = roc_auc_score(test_labels, probabilities)
      metrics['k'] = k
      metrics['pc'] = pc
      metrics['threshold'] = np.round(threshold, 2)

      tuning_table.append(metrics)

      if metrics['f1'] > best_f1:
        best_f1 = metrics['f1']
        best_params = {'k': k, 'pc':pc}
    
    # set best parameters
    self.fit(self.reference_data, best_params['k'], best_params['pc'])

    # reference_embeddings = self.embedder.encode(self.reference_data)
    # self.knn = NearestNeighbors(n_neighbors=best_params['k'], metric='cosine').fit(reference_embeddings)
    # distances, indices = self.knn.kneighbors(reference_embeddings)
    # self.threshold = pd.Series(distances[:,-1]).quantile(best_params['pc'])

    return tuning_table, test_data
  
  def get_params(self):
    return {'k':self.k, 'pc':self.pc, 'threshold':self.threshold}
  
  def set_dynamic_parameters(self, reference_data):
    self.k = int(len(reference_data) * 0.1)

    reference_embeddings = self.embedder.encode(reference_data)
    self.knn = NearestNeighbors(n_neighbors=self.k, metric='cosine').fit(reference_embeddings)

    distances, indices = self.knn.kneighbors(reference_embeddings)
    dist_k = distances[:,-1]
    self.threshold = np.mean(dist_k) + 2.5 * np.std(dist_k)

    return self

  def predict(self, test_queries: List[str], cache=True):
    """

    Args:
      test_queries: list of queries (str) to run inference on
      update (bool): when True, will update the reference set with test_queries

    Returns:
      predictions ( List[bool]): list of boolean drift predictions
      distances (List[float]): list of distances to the kth nearest neighbor
      queries (List[int]): 2d array of k closest queries to inputs
    """
    test_embeddings = self.embedder.encode(test_queries)
    distances, indices = self.knn.kneighbors(test_embeddings)
    predictions = distances[:,-1] > self.threshold
    probabilities = value_to_probability(distances[:,-1], threshold=self.threshold)

    # nearest_queries = [self.reference_data[x] for x in indices]
    
    # print(indices)
    # print(indices[0])
    # print(self.reference_data[indices[0].astype(int)])
    nearest_queries = []
    for arr in indices:
        nearest_queries.append([self.reference_data[x] for x in arr])
    
    if cache:
      self.query_cache += _to_list(test_queries)
      self.embedding_cache += _to_list(test_embeddings)

    return predictions, probabilities, nearest_queries
  
  def retrain(self):
    if len(self.query_cache) == 0:
      return

    self.reference_data = np.concatenate([self.reference_data, self.query_cache])
    self.query_cache = []

    self.reference_embeddings = np.concatenate([self.reference_embeddings, np.array(self.embedding_cache)])
    self.embedding_cache = []

    self.knn = NearestNeighbors(n_neighbors=self.k, metric='cosine').fit(self.reference_embeddings)
    distances, indices = self.knn.kneighbors(self.reference_embeddings)
    self.threshold = pd.Series(distances[:,-1]).quantile(self.pc)