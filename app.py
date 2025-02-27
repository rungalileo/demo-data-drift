import streamlit as st
import numpy as np

from drift import DriftEstimator
from data import load_verizon_queries, load_airbnb_queries


st.title("Drift Detection Demo")

def populate_verizon_data():
    queries = load_verizon_queries()
    st.session_state.seed_ds = "\n".join(queries)

def populate_airbnb_data():
    queries = load_airbnb_queries()
    st.session_state.seed_ds = "\n".join(queries)

# Sidebar Inputs
# st.sidebar.header("Import Data")
# st.sidebar.button("Load Verizon Data", on_click=populate_verizon_data)
# st.sidebar.button("Load Airbnb Data", on_click=populate_airbnb_data)
# model_name = st.sidebar.selectbox("Choose Embedding Model", ["all-MiniLM-L6-v2", "all-mpnet-base-v2"])
# min_cluster_size = st.sidebar.slider("Min Cluster Size", 2, 20, 5)
# min_samples = st.sidebar.slider("Min Samples", 1, 10, 2)

# User Input Queries
st.header("📌 Reference Queries")
col1, col2, col3 = st.columns(3)
with col1:
    st.button("Use Verizon Dataset", on_click=populate_verizon_data)
with col2:
    st.button("Use Airbnb Dataset", on_click=populate_airbnb_data)

user_queries = st.text_area("Enter queries (one per line)", height=150, key="seed_ds").split("\n")
user_queries = [q.strip() for q in user_queries if q.strip()]
st.session_state.user_queries = user_queries



st.session_state.new_queries = []
st.session_state.new_embeddings = []
st.session_state.new_cluster_labels = []

if st.button("Submit 🚀", key='submit') and user_queries:

    st.text("Processing reference dataset...")

    drift = DriftEstimator(
        dim=20,
        min_cluster_size=min(len(user_queries), 10),
        min_samples=5,
    )

    st.text("Fitting drift model....")

    drift.fit(user_queries)

    st.session_state.drift = drift

    # # Compute embeddings & cluster
    # embeddings = generate_embeddings(user_queries, model_name)
    # cluster_labels, clusterer = cluster_queries(embeddings, min_cluster_size, min_samples)
    
    # # Reduce dimensions for visualization
    # reduced_embeddings = reduce_dimensionality(embeddings)
    
    # # Store data in session state
    # st.session_state.embeddings = embeddings
    # st.session_state.cluster_labels = cluster_labels
    # st.session_state.reduced_embeddings = reduced_embeddings
    # st.session_state.queries = user_queries
    # st.session_state.clusterer = clusterer
    

# -------------------------------
# Plot Initial Clusters
# -------------------------------
if "drift" in st.session_state:

    st.text(f"Found {len(st.session_state.drift.clusters)} query types in {len(st.session_state.user_queries)} submitted queries")

    # st.subheader("📊 Cluster Visualization")
    
    # # Prepare DataFrame for Plotly
    # df = pd.DataFrame(st.session_state.reduced_embeddings, columns=["x", "y"])
    # df["Query"] = st.session_state.queries
    # df["Cluster"] = st.session_state.cluster_labels.astype(str)

    # # Scatter plot
    # fig = px.scatter(df, x="x", y="y", color="Cluster", hover_data=["Query"], title="HDBSCAN Clustering")
    # st.plotly_chart(fig, use_container_width=True)

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
            st.success(f"Probability of Drift = {p_drift}")
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
