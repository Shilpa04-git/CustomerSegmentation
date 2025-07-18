import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("C:/shilpa/Mall_Customers.csv")
df.info()
df.describe()
plt.figure(figsize=(8, 6))
sns.histplot(df['Annual Income (k$)'], bins=20, kde=True)
plt.title("Distribution of Annual Income")
plt.show()
plt.figure(figsize=(8, 6))
sns.histplot(df['Spending Score (1-100)'], bins=20, kde=True)
plt.title("Distribution of Spending Score")
plt.show()
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df)
plt.title("Income vs Spending Score")
plt.show()

st.title("🛍️ Customer Segmentation")

df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
wcss=[]
for i in range (1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    print(f"WCSS for k={i}: {kmeans.inertia_}")
plt.figure(figsize=(8,6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elobow method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.grid(True)
plt.show()
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
print(df)

plt.figure(figsize=(8,6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster', data=df, palette='Set1')
plt.title("Customer Segments")
plt.legend()
plt.show()
cluster_summary = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
print("\nCluster Summary:")
print(cluster_summary)
df.to_csv("Segmented_Customers.csv", index=False)

  
st.subheader("Clustered Customers")
fig, ax = plt.subplots()
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'],
                    hue=df['Cluster'], palette='Set1', ax=ax)
st.pyplot(fig)

if st.checkbox("Show Clustered Data"):
    st.write(df)


st.subheader("📊 Test New Customer Data")

# Input fields in Streamlit
income = st.number_input("Enter Annual Income (k$):", min_value=0, max_value=200, value=50)
score = st.number_input("Enter Spending Score (1-100):", min_value=1, max_value=100, value=50)

if st.button("Predict Cluster"):
    # Step 1: Create array from input
    new_customer = np.array([[income, score]])

    # Step 2: Scale it (use same scaler as training)
    new_scaled = scaler.transform(new_customer)

    # Step 3: Predict cluster
    predicted_cluster = kmeans.predict(new_scaled)[0]

    st.success(f"The predicted cluster for this customer is: **Cluster {predicted_cluster}**")

    # Optional: Show on scatter plot
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'],
                    hue=df['Cluster'], palette='Set1', ax=ax2)
    ax2.scatter(income, score, color='black', s=100, label='New Customer', marker='X')
    ax2.legend()
    st.pyplot(fig2)