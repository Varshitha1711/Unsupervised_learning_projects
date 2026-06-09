import pandas as pd

from sklearn.cluster import KMeans
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

X_scaled = scaler.fit_transform(X)

model = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = model.fit_predict(
    X_scaled
)

score = silhouette_score(
    X_scaled,
    clusters
)

print(
    f"Silhouette Score: {score:.3f}"
)