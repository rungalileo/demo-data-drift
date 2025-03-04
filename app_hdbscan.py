import streamlit as st
import numpy as np
from datasets import load_dataset

from drift_hdbscan import DriftEstimator
from data import load_verizon_queries, load_airbnb_queries

from embeddings import Embedder
from drift import NNDriftEstimator


st.title("Drift Detection Demo")

DEMO_MAX_DS_SIZE = 500

def populate_verizon_data():
    queries = load_verizon_queries()
    st.session_state.seed_ds = "\n".join(queries)

def populate_airbnb_data():
    queries = load_airbnb_queries()
    st.session_state.seed_ds = "\n".join(queries)

def populate_python_data():
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca")
    queries = ds['train']['instruction']
    queries = np.random.choice(queries, DEMO_MAX_DS_SIZE, replace=False)

    st.session_state.seed_ds = "\n".join(queries)

def populate_medical_data():
    ds = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled")
    queries = ds['train']['question']
    queries = np.random.choice(queries, DEMO_MAX_DS_SIZE, replace=False)

    st.session_state.seed_ds = "\n".join(queries)

#
# Load OOD DATA SOURCE
#
ds = load_dataset("clinc/clinc_oos", "small")
OOD_QUERIES = ds['validation']['text']

# Sidebar Inputs
# st.sidebar.header("Import Data")
# st.sidebar.button("Load Verizon Data", on_click=populate_verizon_data)
# st.sidebar.button("Load Airbnb Data", on_click=populate_airbnb_data)
# model_name = st.sidebar.selectbox("Choose Embedding Model", ["all-MiniLM-L6-v2", "all-mpnet-base-v2"])
# min_cluster_size = st.sidebar.slider("Min Cluster Size", 2, 20, 5)
# min_samples = st.sidebar.slider("Min Samples", 1, 10, 2)

# User Input Queries
st.header("📌 Reference Queries")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.button("Use Verizon Dataset", on_click=populate_verizon_data)
with col2:
    st.button("Use Airbnb Dataset", on_click=populate_airbnb_data)
with col3:
    st.button("Use Python Code Gen Dataset", on_click=populate_airbnb_data)
with col4:
    st.button("Use Medical QA Dataset", on_click=populate_airbnb_data)

user_queries = st.text_area("Enter queries (one per line)", height=150, key="seed_ds").split("\n")
user_queries = [q.strip() for q in user_queries if q.strip()]
st.session_state.user_queries = user_queries


st.session_state.new_queries = []
st.session_state.new_embeddings = []
st.session_state.new_cluster_labels = []

if st.button("Submit 🚀", key='submit') and user_queries:

    st.text("Processing reference dataset...")

    drift = NNDriftEstimator(Embedder())
    st.text("Fitting drift model....")
    results, test_data = drift.tune_parameters(user_queries, ood_source=OOD_QUERIES)

    # drift = DriftEstimator(
    #     dim=20,
    #     min_cluster_size=min(len(user_queries), 10),
    #     min_samples=5,
    # )
    
    # drift.fit(user_queries)

    st.session_state.drift = drift
    

# -------------------------------
# Plot Initial Clusters
# -------------------------------
# if "drift" in st.session_state:

#     st.text(f"Found {len(st.session_state.drift.clusters)} query types in {len(st.session_state.user_queries)} submitted queries")

# -------------------------------
# Adding New Queries
# -------------------------------
st.subheader("➕ Add a New Query")
new_query = st.text_input("Enter a new query:")

if st.button("Assign Query"):
    if new_query and "drift" in st.session_state:

        labels, scores, probs = st.session_state.drift.predict([new_query])
        assigned_cluster = labels[0]
        assigned_score = scores[0]
        # p_drift = 1 - probs[0]      
        p_drift = 1 - assigned_score

        print(labels, scores)
        # print(probs)
        #

        # drifting = assigned_cluster == -1 or assigned_score < 0.1

        drifting = False
        if assigned_cluster == -1:
            drifting = True
        else:
            drifting=False

        # if assigned_score < 0.5:
            # cluster_queries = st.session_state.drift.clusters[assigned_cluster]
            # labels, scores, probs = st.session_state.drift.predict(cluster_queries)
            # print("cluster probs")
            # print(probs)
            # min_prob = min(probs)
            # mean_prob = np.mean(probs)
            # if assigned_score < min_prob:
            #     print("DRIFT!")
            #     drifting = True
            # else:
            #     print("NO DRIFT")
            #     drifting=False
            # prob.shape

        # Show Cluster Assignment
        # if assigned_cluster == -1 or assigned_score < 0.5:
        if drifting:
            # st.warning(f"'{new_query}' does not belong to any existing cluster (New Query Type 🚀).")

            st.error(f"'{new_query}' is drifting. Probability of Drift = {p_drift:.2}")
        else:
            qs = "\n\n".join(st.session_state.drift.clusters[assigned_cluster])

            # st.success(f"'{new_query}' assigned to **Cluster {assigned_cluster}** Probability of Drift = {1-assigned_score}.\n\n**Similar Queries**:\n\n{qs}")
            # st.success(f"Probability of Drift = {p_drift}")
            st.success(f"'{new_query}' is NOT drifting.\n\n**Similar Queries**:\n\n{qs}")
        
            # st.success(f"\n\nSimilar Queries:\n\n{qs}")

        

# -------------------------------
# Detecting New Clusters
# -------------------------------
# if len(st.session_state.new_queries) > 2:
#     st.subheader("🔍 Detecting New Clusters")

#     # Merge old + new embeddings and recluster
#     all_embeddings = np.vstack([st.session_state.embeddings] + st.session_state.new_embeddings)
#     all_cluster_labels, new_clusterer = cluster_queries(all_embeddings, min_cluster_size, min_samples)
    

#     # Check if new clusters appeared
#     old_clusters = set(st.session_state.cluster_labels)
#     new_clusters = set(all_cluster_labels)

#     if new_clusters - old_clusters:
#         st.warning(f"🚀 **New Clusters Detected!** Newly formed clusters: {new_clusters - old_clusters}")
    
#     # Update session state
#     print("len QUERIES = ", len(st.session_state.queries))
#     print("len embeddings = ", len(st.session_state.reduced_embeddings))
#     st.session_state.queries += st.session_state.new_queries
#     st.session_state.reduced_embeddings = reduce_dimensionality(all_embeddings)

#     st.session_state.cluster_labels = all_cluster_labels
#     st.session_state.clusterer = new_clusterer
