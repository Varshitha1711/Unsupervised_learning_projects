import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

df = pd.read_csv(
    "data/Wholesale customers data.csv"
)

X = df.drop(
    columns=["Channel", "Region"]
)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(
    eps=1.5,
    min_samples=5
)

dbscan.fit(X_scaled)

joblib.dump(
    dbscan,
    "models/dbscan_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Model Saved Successfully")   