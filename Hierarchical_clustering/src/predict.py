import joblib
import pandas as pd

scaler = joblib.load(
    "models/scaler.pkl"
)

model = joblib.load(
    "models/cluster_predictor.pkl"
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

    return int(cluster)