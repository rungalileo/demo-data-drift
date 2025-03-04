from math import ceil
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoModel
import torch
import umap

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import normalize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_id):
  if "MiniLM" in model_id:
    return SentenceTransformer(model_id)
  else:
    return HFModel(model_id)

class HFModel:
  def __init__(self, model_id):
    self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    self.model.to(DEVICE)
    self.model.eval()

  def _normalize_embeddings(self, emb):
    norm = np.linalg.norm(emb)
    if norm == 0:
       return emb
    return emb / norm

  def encode(self, queries, batch_size=32, mode="mean", normalize_embeddings=True, **kwargs):
    self.model.eval()

    batch_size = batch_size or len(queries)
    batch_start, n_batches = 0, ceil(len(queries) / batch_size)

    outputs = []
    for i_batch in tqdm(range(n_batches)):
      batch_texts = queries[batch_start: batch_start + batch_size]
      batch_start = batch_start + batch_size

      inp = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
      inp = {k:v.to(DEVICE) for k, v in inp.items()}

      with torch.no_grad():
        # output = model.encode(batch_texts, convert_to_tensor=True) # This is not good since 1) it truncates at max_length, 2) some models normalize the embeddings
        output = self.model(**inp)

      if mode == "mean":
          vectors = output.last_hidden_state * inp["attention_mask"].unsqueeze(2)
          vectors = vectors.sum(dim=1) / inp["attention_mask"].sum(dim=-1).view(-1, 1)
      elif mode == "cls":
          vectors = output.last_hidden_state[:, 0, :]
      else:
        raise ValueError(f"We don't support mode {mode} for aggregating embeddings")

      outputs.append(vectors.cpu().numpy())

      del output, vectors

    embeddings = np.concatenate(outputs, axis=0)

    if normalize_embeddings:
      embeddings = self._normalize_embeddings(embeddings)

    return embeddings

class Embedder:
  def __init__(
      self,
      model_name='all-MiniLM-L6-v2',
      reducer=None,
      dim=100,
      dev_mode=False,
  ):
    self.dim = dim
    self.model = get_model(model_name)
    self.dev_mode = dev_mode

    self.reducer_type = reducer
    self.reducer = None

  def _initialize_reducer(self):
    if not self.reducer:
      if not self.reducer_type or not self.dim:
        pass
      # Initialize and fit reducer
      elif self.reducer_type == "parametric_umap":
        self.reducer = umap.parametric_umap.ParametricUMAP(
            n_components=self.dim,
            metric="cosine",
            random_state=42,
            init="random"
          )
      elif self.reducer_type == "umap":
        # this is faster, but not good for projecting new points into the same reduced space
        self.reducer = umap.UMAP(
            n_components=self.dim,
            metric="cosine",
            random_state=42,
            init="random"
          )
      elif self.reducer_type == "pca":
        self.reducer = PCA(n_components=self.dim)
      elif self.reducer_type=="svd":
        self.reducer = TruncatedSVD(n_components=self.dim, random_state=42)
      else:
        raise ValueError(f"Unknown reducer type: {self.reducer_type}")


  def encode(self, queries):
    """Generate sentence embeddings for user queries."""

    embeddings = self.model.encode(queries, convert_to_numpy=True, show_progress_bar=self.dev_mode, normalize_embeddings=True)
    # print(embeddings)

    if not self.reducer and self.reducer_type: # not itialized yet
      self._initialize_reducer()
      embeddings = self.reducer.fit_transform(embeddings)

    elif self.reducer_type:
      # apply reducer
      embeddings = self.reducer.transform(embeddings)

    return embeddings