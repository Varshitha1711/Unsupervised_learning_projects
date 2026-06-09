import streamlit as st
import pandas as pd
import plotly.express as px

from src.predict import predict_cluster

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

st.title(
    "📊 Hierarchical Customer Segmentation"
)

st.markdown(
    """
    Segment customers using Hierarchical Clustering.
    """
)

col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Age",
        18,
        80,
        30
    )

    income = st.slider(
        "Annual Income (k$)",
        1,
        150,
        50
    )

with col2:

    spending = st.slider(
        "Spending Score",
        1,
        100,
        50
    )

if st.button(
    "Predict Customer Segment"
):

    features = {
        "Age": age,
        "Annual Income (k$)": income,
        "Spending Score (1-100)": spending
    }

    cluster = predict_cluster(
        features
    )

    cluster_names = {
        0: "Premium Customers",
        1: "Budget Customers",
        2: "Luxury Shoppers",
        3: "Young Spenders",
        4: "Regular Customers"
    }

    st.success(
        f"Predicted Segment: {cluster_names[cluster]}"
    )

try:

    df = pd.read_csv(
        "data/Mall_Customers.csv"
    )

    fig = px.scatter(
        df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color="Age",
        title="Customer Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

except:
    pass