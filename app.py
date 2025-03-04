import streamlit as st
import numpy as np
from datasets import load_dataset

from drift_hdbscan import DriftEstimator
from data import load_verizon_queries, load_airbnb_queries, load_ood_queries, load_python_code_queries, load_mental_health_queries


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
    queries = load_python_code_queries()
    st.session_state.seed_ds = "\n".join(queries)

def populate_mental_health_data():
    queries = load_mental_health_queries()
    st.session_state.seed_ds = "\n".join(queries)

#
# Load OOD DATA SOURCE
#
OOD_QUERIES = load_ood_queries()


# User Input Queries
st.header("📌 Reference Queries")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.button("Use Verizon Dataset", on_click=populate_verizon_data)
with col2:
    st.button("Use Airbnb Dataset", on_click=populate_airbnb_data)
with col3:
    st.button("Python Code Gen Dataset", on_click=populate_python_data)
with col4:
    st.button("Mental Health QA Dataset", on_click=populate_mental_health_data)

user_queries = st.text_area("Enter queries (one per line)", height=150, key="seed_ds").split("\n")
user_queries = [q.strip() for q in user_queries if q.strip()]
st.session_state.user_queries = user_queries


st.session_state.new_queries = []
st.session_state.new_embeddings = []
st.session_state.new_cluster_labels = []

if st.button("Fit Dataset 🚀", key='submit') and user_queries:

    st.text(f"Processing reference dataset with {len(user_queries)} rows...")

    drift = NNDriftEstimator(Embedder())
    st.text("Fitting drift model....")
    results, test_data = drift.tune_parameters(user_queries, ood_source=OOD_QUERIES)

    st.session_state.drift = drift
    
def refit_drift():
    st.session_state.drift.retrain()
    st.text("Drift model updated!")
# -------------------------------
# Adding New Queries
# -------------------------------
st.subheader("➕ Add a New Query")
new_query = st.text_input("Enter a new query:")

if st.button("Assign Query"):
    if new_query and "drift" in st.session_state:
        
        preds, probs, neighbors = st.session_state.drift.predict([new_query])
        
        drifting = preds[0]
        p_drift = probs[0]
       

        if drifting:
            st.error(f"'{new_query}' is drifting. Probability of Drift = {p_drift:.2}")
        else:
            # qs = "\n\n".join(st.session_state.drift.clusters[assigned_cluster])
            qs = "\n\n".join(neighbors[0])

            # st.success(f"'{new_query}' assigned to **Cluster {assigned_cluster}** Probability of Drift = {1-assigned_score}.\n\n**Similar Queries**:\n\n{qs}")
            # st.success(f"Probability of Drift = {p_drift}")
            st.success(f"'{new_query}' is NOT drifting. Probability of Drift = {p_drift:.2} \n\n**Similar Queries**:\n\n{qs}")


        if len(st.session_state.drift.query_cache) >= st.session_state.drift.k:
            st.button("Update Reference Dataset and Drift Model?", on_click=refit_drift)