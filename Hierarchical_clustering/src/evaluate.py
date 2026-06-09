import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(
    "data/Mall_Customers.csv"
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

model = AgglomerativeClustering(
    n_clusters=5,
    linkage="ward"
)

labels = model.fit_predict(
    X_scaled
)

score = silhouette_score(
    X_scaled,
    labels
)

print(
    f"Silhouette Score: {score:.4f}"
)