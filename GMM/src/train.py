import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

df = pd.read_csv(
    "GMM/data/Mall_Customers.csv"
)

X = df[
    [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

gmm = GaussianMixture(
    n_components=5,
    random_state=42
)

clusters = gmm.fit_predict(
    X_scaled
)

df["Cluster"] = clusters

profiles = (
    df.groupby("Cluster")
      .mean(numeric_only=True)
)

joblib.dump(
    gmm,
    "GMM/models/gmm_model.pkl"
)

joblib.dump(
    scaler,
    "GMM/models/scaler.pkl"
)

joblib.dump(
    profiles,
    "GMM/models/cluster_profiles.pkl"
)

print("Model Saved Successfully")