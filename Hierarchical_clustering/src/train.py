import pandas as pd
import joblib

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(
    "Hierarchical_clustering/data/Mall_Customers.csv"
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

hierarchical = AgglomerativeClustering(
    n_clusters=5,
    linkage="ward"
)

labels = hierarchical.fit_predict(
    X_scaled
)

knn = KNeighborsClassifier(
    n_neighbors=5
)

knn.fit(
    X_scaled,
    labels
)

joblib.dump(
    scaler,
    "Hierarchical_clustering/models/scaler.pkl"
)

joblib.dump(
    knn,
    "Hierarchical_clustering/models/cluster_predictor.pkl"
)

print(
    "Model training completed."
)