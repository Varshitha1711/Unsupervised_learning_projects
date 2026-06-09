import joblib
import pandas as pd

model = joblib.load(
    "models/kmeans_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

def predict_cluster(features):

    df = pd.DataFrame(
        [features]
    )

    scaled = scaler.transform(
        df
    )

    cluster = model.predict(
        scaled
    )[0]

    return cluster