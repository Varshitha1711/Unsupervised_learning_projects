import streamlit as st
import plotly.express as px
import pandas as pd

from src.predict import predict_customer

st.set_page_config(
    page_title="GMM Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

st.title(
    "🛍️ Customer Segmentation using GMM"
)

st.write(
    """
    Enter customer details and discover
    which customer segment they belong to.
    """
)

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

with col2:

    income = st.number_input(
        "Annual Income (k$)",
        min_value=1,
        max_value=200,
        value=60
    )

with col3:

    score = st.number_input(
        "Spending Score",
        min_value=1,
        max_value=100,
        value=50
    )

if st.button(
    "Find Segment"
):

    features = {
        "Age": age,
        "Annual Income (k$)": income,
        "Spending Score (1-100)": score
    }

    cluster, segment, prob = predict_customer(
        features
    )

    st.success(
        f"Segment: {segment}"
    )

    st.metric(
        "Confidence",
        f"{prob*100:.2f}%"
    )

st.divider()

st.subheader(
    "Example Customer Segments"
)

demo = pd.DataFrame(
    {
        "Income":[20,40,80,100,120],
        "Score":[20,50,80,90,95],
        "Segment":[
            "Budget",
            "Average",
            "Premium",
            "High Spender",
            "Luxury"
        ]
    }
)

fig = px.scatter(
    demo,
    x="Income",
    y="Score",
    color="Segment",
    size="Income"
)

st.plotly_chart(
    fig,
    use_container_width=True
)