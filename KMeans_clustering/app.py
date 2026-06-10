import streamlit as st

from src.predict import predict_cluster

st.title(
    "Customer Segmentation using K-Means"
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100
)

income = st.number_input(
    "Annual Income (k$)",
    min_value=1
)

score = st.slider(
    "Spending Score",
    1,
    100
)

if st.button(
    "Predict Segment"
):

    features = {

        "Age": age,

        "Annual Income (k$)": income,

        "Spending Score (1-100)": score
    }

    cluster = predict_cluster(
        features
    )

    st.success(
        f"Customer belongs to Cluster {cluster}"
    )