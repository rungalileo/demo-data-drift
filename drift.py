import hdbscan
from sentence_transformers import SentenceTransformer
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
import umap
import seaborn as sns
import numpy as np

class DriftEstimator:
  def __init__(
    self,
    dim=4,
    min_cluster_size=10,
    min_samples=3,
    model_name='all-MiniLM-L6-v2',
    parametric_umap=True,
  ):
    self.dim = dim
    self.min_cluster_size = min_cluster_size
    self.min_samples = min_samples
    self.parametric_umap = parametric_umap

    self.model = SentenceTransformer(model_name)

    self.reducer = None

    self.clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='l2',
        prediction_data=True,
        gen_min_span_tree=True
    )

  def get_params(self,deep=True):
    return {'dim':self.dim,'min_cluster_size':self.min_cluster_size,'min_samples':self.min_samples}

  def set_params(self, **params):
    for param, value in params.items():
        setattr(self, param, value)  # Update existing attributes

    self.clusterer = hdbscan.HDBSCAN(
        min_cluster_size=self.min_cluster_size,
        min_samples=self.min_samples,
        metric='l2',
        prediction_data=True,
        gen_min_span_tree=True
    )

    return self
  
  def _generate_cluster_membership_dict(self, queries):
    self.clusters = {}
    for i, label in enumerate(self.clusterer.labels_):
      if label not in self.clusters:
        self.clusters[label] = []

      self.clusters[label].append(queries[i])
  
  def print_cluster_membership(self):
    for key in sorted(self.clusters):
      print("\n", key, len(self.clusters[key]))
      for query in self.clusters[key]:
        print(query)
  
  def fit(self, queries):
    self.reducer = None # reset reducer
    self.queries = queries
    self.embeddings = self._generate_embeddings(queries)
    self.clusterer.fit(self.embeddings)

    # store cluster dict
    self._generate_cluster_membership_dict(queries)

  # def fit_predict(self, queries):
  #   embeddings = self._generate_embeddings(queries)
  #   self.clusterer.fit_predict(embeddings)

  #   # store cluster dict
  #   self._generate_cluster_membership_dict(queries)


  def score(self, data, targets=None):
    # embeddings = self._generate_embeddings(data)
    # self.clusterer.fit(embeddings)

    if hasattr(self.clusterer, "relative_validity_"):
        return self.clusterer.relative_validity_
    else:
        return -1  # Return a default low score if clustering failed or is not done yet

  def predict(self, new_queries):
    # return hdbscan.prediction.membership_vector(self.clusterer, self._generate_embeddings(queries))
    embeddings = self._generate_embeddings(new_queries)

    soft_labels = hdbscan.prediction.membership_vector(self.clusterer, embeddings)
    # print(soft_labels)
    soft_probs = np.max(soft_labels, axis=-1)

    labels, scores = hdbscan.approximate_predict(self.clusterer, embeddings)

    return labels, scores, soft_probs

#   def visualize_tsne(self, queries):
#     embeddings = self._generate_embeddings(queries)

#     # reduce with TSNE
#     perplexity = min(30, embeddings.shape[0]-1)
#     tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
#     reduced_embeddings = tsne.fit_transform(embeddings)

#     # plot
#     plt.figure(figsize=(10, 6))
#     unique_labels = set(self.clusterer.labels_)
#     for label in unique_labels:
#         indices = self.clusterer.labels_ == label
#         plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=f'Cluster {label}')

#     plt.legend()
#     plt.title("HDBSCAN Clustering of Queries with TSNE")
#     plt.show()

  def visualize_umap(self, queries):
    embeddings = self._generate_embeddings(queries)

    # Visualize Clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=embeddings[:,0], y=embeddings[:,1], hue=self.clusterer.labels_, palette="bright", legend="full", s=30)
    plt.title("HDBSCAN Clusters in UMAP Space")
    plt.show()

  def _generate_embeddings(self, queries):
    """Generate sentence embeddings for user queries."""

    embeddings = self.model.encode(queries, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)

    if not self.reducer:
      # Initialize and fit reducer
      if self.parametric_umap:
        self.reducer = umap.parametric_umap.ParametricUMAP(
            n_components=self.dim,
            metric="cosine",
            random_state=42,
            init="random"
          )
      else:
        # this is faster, but not good for projecting new points into the same reduced space
        self.reducer = umap.UMAP(
            n_components=self.dim,
            metric="cosine",
            random_state=42,
            init="random"
          )
      reduced_embeddings = self.reducer.fit_transform(embeddings)

    else:
      # apply reducer
      reduced_embeddings = self.reducer.transform(embeddings)

    return reduced_embeddings