import streamlit as st

from src.predict import predict_customer
st.set_page_config(
    page_title="Customer Anomaly Detection",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.big-font {
    font-size:22px !important;
    font-weight:bold;
}
</style>
""",
unsafe_allow_html=True)

st.markdown(
    '<p class="big-font">Wholesale Customer Analysis</p>',
    unsafe_allow_html=True
)


st.title(
    "📊 Customer Anomaly Detection using DBSCAN"
)

st.write(
    "Detect unusual customer purchasing behavior."
)

col1, col2 = st.columns(2)

with col1:

    fresh = st.number_input(
        "Fresh",
        min_value=0
    )

    milk = st.number_input(
        "Milk",
        min_value=0
    )

    grocery = st.number_input(
        "Grocery",
        min_value=0
    )

with col2:

    frozen = st.number_input(
        "Frozen",
        min_value=0
    )

    detergents = st.number_input(
        "Detergents Paper",
        min_value=0
    )

    delicassen = st.number_input(
        "Delicassen",
        min_value=0
    )

if st.button(
    "Analyze Customer"
):

    features = {
        "Fresh": fresh,
        "Milk": milk,
        "Grocery": grocery,
        "Frozen": frozen,
        "Detergents_Paper": detergents,
        "Delicassen": delicassen
    }

    result = predict_customer(
        features
    )

    if result == "Anomaly":

        st.error(
            "⚠ Potential Anomalous Customer"
        )

    else:

        st.success(
            "✅ Normal Customer"
        )