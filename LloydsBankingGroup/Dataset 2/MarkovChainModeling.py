import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load the monthly RFM dataset
df = pd.read_csv("E:/mini-project/mini/dsmp-2024-groupm17/monthly_rfm_cal.csv")

# 2. Function to build a Markov transition matrix
def build_transition_matrix(from_labels, to_labels, n_clusters=6):
    matrix = np.zeros((n_clusters, n_clusters))  # Initialize the transition matrix
    for i in range(len(from_labels)):
        from_c = int(from_labels.iloc[i])  # Starting state
        to_c = int(to_labels.iloc[i])      # Destination state
        matrix[from_c][to_c] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)  # Sum of each row
    return (matrix / (row_sums + 1e-6)).round(4)  # Normalize and return

# 3. Markov Chain Forecast (Method B: No usage of M4 labels)
m1_dist = df['Cluster_M1'].value_counts(normalize=True).sort_index().values  # Initial distribution from M1
actual_m4 = df['Cluster_M4'].value_counts(normalize=True).sort_index().values  # Actual distribution in M4

T_12 = build_transition_matrix(df['Cluster_M1'], df['Cluster_M2'])  # Transition from M1 to M2
T_23 = build_transition_matrix(df['Cluster_M2'], df['Cluster_M3'])  # Transition from M2 to M3
predicted_m4_chain = m1_dist @ T_12 @ T_23 @ T_23  # Forecast M4 distribution via Markov chain

# Output comparison of predicted vs actual M4 distribution
comparison = pd.DataFrame({
    'Cluster': range(6),
    'Predicted_M4': predicted_m4_chain,
    'Actual_M4': actual_m4
}).round(4)
print(comparison)

# 4. Elbow Method plot to determine optimal number of clusters
rfm = df[['Recency_M3', 'Frequency_M3', 'Monetary_M3']].dropna()  # Use RFM features from M3
rfm_scaled = StandardScaler().fit_transform(rfm)  # Standardize the features

inertias = []
for k in range(1, 11):  # Try 1 to 10 clusters
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(rfm_scaled)
    inertias.append(model.inertia_)  # Record inertia for each k

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.tight_layout()
plt.savefig("Elbow Method for Optimal K.png")  # Save the plot
plt.show()

# 5. PCA-based visualization of KMeans clustering
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)  # Reduce dimensions for visualization
kmeans_model = KMeans(n_clusters=6, random_state=42, n_init=10)
labels = kmeans_model.fit_predict(rfm_scaled)  # Perform clustering

plt.figure(figsize=(8, 6))
sns.scatterplot(x=rfm_pca[:, 0], y=rfm_pca[:, 1], hue=labels, palette='Set2', s=50)
plt.title('KMeans Clustering (PCA Projection)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig("KMeans Clustering (PCA Projection).png")  # Save the plot
plt.show()

# 6. Bar chart: Predicted vs Actual distribution of M4 clusters
bar_width = 0.35
index = np.arange(6)

plt.figure(figsize=(10, 6))
plt.bar(index, actual_m4, bar_width, label='Actual M4')  # Actual values
plt.bar(index + bar_width, predicted_m4_chain, bar_width, label='Predicted M4')  # Predicted values
plt.xlabel('Cluster')
plt.ylabel('Proportion')
plt.title('Prediction vs Actual (Markov Chain Forecast)')
plt.xticks(index + bar_width / 2, [f'Cluster {i}' for i in range(6)])
plt.legend()
plt.tight_layout()
plt.savefig("Prediction vs Actual (Markov Chain Forecast).png")  # Save the plot
plt.show()

# 7. Bar chart showing absolute prediction error for each cluster
errors = np.abs(actual_m4 - predicted_m4_chain)  # Calculate absolute errors

plt.figure(figsize=(8, 5))
plt.bar(range(6), errors)
plt.xlabel('Cluster')
plt.ylabel('Absolute Error')
plt.title('Prediction Error per Cluster')
plt.xticks(range(6), [f'Cluster {i}' for i in range(6)])
plt.tight_layout()
plt.savefig("Prediction Error per Cluster.png")  # Save the plot
plt.show()
