import joblib
import pandas as pd

model = joblib.load(
    "GMM/models/gmm_model.pkl"
)

scaler = joblib.load(
    "GMM/models/scaler.pkl"
)

cluster_profiles = joblib.load(
    "GMM/models/cluster_profiles.pkl"
)


def get_cluster_name(cluster):

    names = {
        0: "Premium Customers",
        1: "Budget Customers",
        2: "Luxury Customers",
        3: "Average Customers",
        4: "High Spenders"
    }

    return names.get(
        cluster,
        "Customer Segment"
    )


def predict_customer(features):

    df = pd.DataFrame(
        [features]
    )

    scaled = scaler.transform(df)

    cluster = model.predict(
        scaled
    )[0]

    probability = (
        model.predict_proba(
            scaled
        ).max()
    )

    return (
        cluster,
        get_cluster_name(cluster),
        probability
    )