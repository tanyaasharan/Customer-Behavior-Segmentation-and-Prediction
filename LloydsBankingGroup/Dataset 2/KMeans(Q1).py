import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the cleaned dataset
df = pd.read_csv("E:/mini-project/mini/dsmp-2024-groupm17/cleaned_dataset2.csv")
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

# Drop rows where DateTime could not be parsed
df = df.dropna(subset=['DateTime'])

# Extract the month from DateTime
df['Month'] = df['DateTime'].dt.month

# Function: Calculate RFM metrics for a given month
def calculate_monthly_rfm(df_month):
    ref_date = df_month['DateTime'].max()
    rfm = df_month.groupby('Account No').agg({
        'DateTime': lambda x: (ref_date - x.max()).days,  # Days since last transaction
        'Amount': ['count', 'sum']  # Number of transactions and total amount
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm.reset_index()

# Function: Apply clustering to monthly RFM data
def cluster_rfm(rfm, n_clusters=6):
    scaler = StandardScaler()  # Standardize the data
    X = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  # KMeans clustering
    rfm['Cluster'] = model.fit_predict(X)  # Assign cluster labels
    return rfm  # Return RFM data with cluster labels

# Main loop: Calculate RFM and apply clustering for each month
monthly_results = {}

for month in [1, 2, 3, 4]:
    df_month = df[df['Month'] == month]  # Filter data for the current month
    rfm_month = calculate_monthly_rfm(df_month)  # Compute RFM metrics
    clustered = cluster_rfm(rfm_month)  # Perform clustering
    renamed = clustered.rename(columns={  # Rename columns with month suffix
        'Recency': f'Recency_M{month}',
        'Frequency': f'Frequency_M{month}',
        'Monetary': f'Monetary_M{month}',
        'Cluster': f'Cluster_M{month}'
    })
    monthly_results[month] = renamed  # Save the result

# Merge RFM results from each month
merged_full = monthly_results[1]
for month in [2, 3, 4]:
    merged_full = merged_full.merge(monthly_results[month], on='Account No', how='inner')

# Save the merged results to a CSV file
output_path_full = "E:/mini-project/mini/dsmp-2024-groupm17/monthly_rfm_cal.csv"
merged_full.to_csv(output_path_full, index=False)

output_path_full
