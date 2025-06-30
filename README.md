# Customer-Behavior-Segmentation-and-Prediction
Analyzed Lloyds' transaction data using RFM &amp; K-Means for customer segmentation. Applied Markov Chains to predict behavioral shifts. Aimed to identify at-risk customers and personalise financial support.

## Project Overview
This project explores the use of transactional data to understand customer spending patterns, segment customers into distinct behavioral groups, and predict their future behavior. Leveraging a synthetic dataset modeled on real customer interactions, we demonstrate how data science techniques can transform raw transaction logs into actionable insights for retail banks. The primary goal is to proactively support customers by identifying potential financial difficulties and personalizing services.

Initially, Lloyds Banking Group provided two datasets. We opted for the second dataset, simulated_fake_transactions_dataset, due to the presence of comprehensive timestamp information, which was crucial for our time-series analysis and the calculation of Recency, Frequency, and Monetary (RFM) metrics.

## Data 
The dataset used, **simulated_fake_transactions_dataset**, contains individual transactional activities of Lloyds Banking Group customers. Each row represents a single transaction and includes the following key details:
1. **Date and Timestamp:** Date and time of the transaction. 
2. **Amount:** The monetary value of the transaction. 
3. **Balance:** The account balance after the transaction. 
4. **Third Party Account No / Third Party Name:** Identifiers for the recipient of the transaction. 
5. **merchant_category_group:** High-level labels categorizing transactions (e.g., Retail & Fashion, Supermarkets).

## Methodology 

<img width="210" alt="Screenshot 2025-06-30 at 2 12 01 AM" src="https://github.com/user-attachments/assets/b37c1afc-b673-4b5e-8677-5ff5cfc197e9" />


### Data Cleaning and Preprocessing
A robust data cleaning and preprocessing pipeline was essential to ensure data quality and consistency:
1. **Standardizing Date and Time:** The **Date** and **Timestamp** columns were merged into a single **DateTime** column for easier time-based aggregation and sorting.
3. **Handling Third-Party Inconsistencies:**
   - Transactions with a **Third Party Name** but missing **Third Party Account No** were labeled as **MTrx** (merchant transactions).
   - Transactions with a **Third Party Account No** but missing **Third Party Name** were labeled as **P2P** (peer-to-peer transfers).
   - Transactions missing both fields were labeled as **Unknown**.
4. **Addressing Missing Values:** Rows missing all of Account No, Amount, and Balance were dropped.
5. **Standardizing Data Types:**
   - Amount and Balance were converted to floating-point numbers to support numerical operations.
   - Account No was stored as a string to prevent formatting issues.
   - The merged DateTime column was converted to a proper datetime object for consistent time-based analysis.

### Exploratory Data Analysis (EDA) and Feature Engineering
EDA was conducted to understand customer behavioral characteristics, guiding feature selection for segmentation and modeling.
1. **Transaction Frequency:** Calculated as the total number of transactions per customer within monthly intervals.
   
2. **Monetary Behavior:** Measured by aggregating debit transactions (negative amounts) to represent total monthly spending.
   
3. **Transaction Categories:** merchant_category_group was used to categorize spending (e.g., Retail & Fashion, Supermarkets). Low-frequency categories were grouped into "Other" to reduce noise.
<img width="468" alt="Screenshot 2025-06-30 at 2 15 21 AM" src="https://github.com/user-attachments/assets/9675d1aa-913a-428b-a5a8-33bd674eaa87" />

4. **Daily Spending Patterns:** Observed recurring spikes at the beginning of each month, likely corresponding to salary deposits or monthly subscriptions, highlighting the importance of Recency and Frequency features.
<img width="418" alt="Screenshot 2025-06-30 at 2 21 30 AM" src="https://github.com/user-attachments/assets/f36d3875-2ff4-4ce5-bbad-65a5baab0254" />

   
5. **Distribution of Transaction Amounts:** Showed a near-normal spread, indicating routine financial activity, with minimal extreme values.
<img width="470" alt="Screenshot 2025-06-30 at 2 14 50 AM" src="https://github.com/user-attachments/assets/a2b74df8-c356-4d9b-acfb-68be2c1af380" />

### RFM (Recency, Frequency, Monetary) Modeling

The RFM model was applied to quantify customer engagement and financial behavior. 
1. **Recency:** Number of days since a customer's last transaction.
2. **Frequency:** Total number of transactions made during a specific period (e.g., monthly).
3. **Monetary:** Sum of all spending (total value of negative transactions).

Initially, RFM values were calculated using the entire year's data for a holistic view. However, for predictive modeling with Markov Chains, monthly RFM calculations for the first quarter (January to March) were performed to track behavioral evolution over time.

### K-Means Clustering for Customer Segmentation

K-Means, an unsupervised machine learning algorithm, was used to group customers into distinct behavioral segments based on their RFM values.
1. **Feature Normalization:** RFM features were normalized using Min-Max scaling ```(X_scaled=fracX−X_minX_max−X_min)``` to ensure equal contribution to distance calculations and prevent monetary values from dominating the clustering.
2. **Optimal Number of Clusters (K):** The Elbow Method was employed by plotting the Within-Cluster Sum of Squares (WCSS) against different K values. An elbow was observed at K=6, indicating the optimal number of clusters.
   <img width="449" alt="Screenshot 2025-06-30 at 2 10 15 AM" src="https://github.com/user-attachments/assets/f88ecda1-1f25-45fe-8418-6ee421297e46" />
3. **Cluster Interpretation:** Each of the 6 clusters represented a distinct customer profile.
   <img width="465" alt="Screenshot 2025-06-30 at 2 10 51 AM" src="https://github.com/user-attachments/assets/dd2c8081-4307-4afd-8b05-311d6ee595d9" />

### Markov Chain Model for Behavioral Prediction
To understand and predict how customer behavior changes over time, a first-order Markov Chain model was implemented. 
1. **States:** Each behavioral cluster derived from K-Means was considered a "state."
2. **Transition Matrix (P):** Constructed using actual cluster labels from January to February and February to March. Each element P_ij represents the probability of a customer transitioning from Cluster i to Cluster j.


```P_ij = Total customers in Cluster i/ Customers transitioning from Cluster i to j```

3. **Prediction:** The model predicted the cluster distribution for April (Month 4) by multiplying the March (Month 3) cluster distribution state vector ```(X_t)``` with the transition matrix ```(P): X_t+1=X_tP```.
4. **Evaluation:** The predicted distribution for April was compared against the actual observed distribution.
   
​	<img width="275" alt="Screenshot 2025-06-30 at 2 19 54 AM" src="https://github.com/user-attachments/assets/b59afa15-6ada-41a9-aaf1-6a616b6edf92" />

## Future Work and Improvements
Several enhancements can improve the model's accuracy and insights:

1. Credit-to-Debit Ratios: Incorporating this ratio, especially with more detailed income data, could provide deeper insights into financial stability and potential overdraft risk. 
2. Overdraft Analysis: Further exploration of overdraft frequency, duration, and recovery patterns using time-series or anomaly detection models. 
3. Additional Features: Integrating credit scores, overdraft history, and income patterns to reflect real-world banking behavior more closely. 
4. Higher-Order Markov Chains or Hidden Markov Models (HMMs): To account for longer behavioral timelines and more context-aware transitions.
5. Financial Stability Metrics: Computing the standard deviation of account balance over ```time (sigma(textBalance_i(t)))``` to quantify customer financial stability.

## How to Cite
If you use this work in your research, please cite:
```
@software{tanyaasharan2025customerbehavior,
  author = {Babare, Shreya and Liu, Yupeng and Sharan, Tanya and Zhang, Jiaqi},
  title = {Customer-Behavior-Segmentation-and-Prediction},
  year = {2025},
  url = {https://github.com/tanyaasharan/Customer-Behavior-Segmentation-and-Prediction}
}
```



 



   
