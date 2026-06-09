import joblib
import pandas as pd

model = joblib.load(
    "models/dbscan_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)


def predict_customer(features):

    df = pd.DataFrame(
        [features]
    )

    scaled = scaler.transform(df)

    neighbors = model.fit_predict(
        scaled
    )

    label = neighbors[0]

    if label == -1:
        return "Anomaly"

    return "Normal"