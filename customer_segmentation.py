# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


customer_data = pd.read_csv('customer_data.csv')

# Display the first few rows of the dataset to ensure it's loaded correctly
print(customer_data.head())

# Data Preprocessing (Feature scaling)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data[['Age', 'Spending_Score']])

# KMeans Clustering (Customer Segmentation)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(scaled_data)

# Add the cluster labels to the original data
customer_data['cluster'] = kmeans.labels_

# Visualize the clusters
plt.figure(figsize=(8,6))
sns.scatterplot(data=customer_data, x='Age', y='Spending_Score', hue='cluster', palette='Set1')
plt.title("Customer Segmentation using KMeans")
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.show()
