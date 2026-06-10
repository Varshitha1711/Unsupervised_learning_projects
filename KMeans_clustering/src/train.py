import pandas as pd
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(
    "KMeans_clustering/data/Mall_Customers.csv"
)

X = df[
    [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

kmeans.fit(
    X_scaled
)

joblib.dump(
    scaler,
    "KMeans_clustering/models/scaler.pkl"
)

joblib.dump(
    kmeans,
    "KMeans_clustering/models/kmeans_model.pkl"
)

print(
    "Model Saved Successfully"
)