import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
sns.set(context="notebook",palette="Spectral",style='darkgrid', front_scale=1.5, color_codes=True)
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv("Mall_Customers.csv") 
df.head()
df.info()
df.describe()

#Verifying missing values
df.isnull().sum()
df.shape

#Using only Age and Income variables for easy visualisation
X=df.iloc[:,[2,3]].values
df.iloc[:,[2,3]].head()


#Using the elbow method to find the optimal number of clusters 
from sklearn.cluster import KMeans
#Whitin cluster sum square, It is defined as the sum of square distances between the centroids and each points.
wcss=[]
for i in range (1,30):
    #init='kmeans++':selects initial clusters centroids sampling based on an empirical probability of the points
    #centroids are selected where there are more points
    #One can use "random" initialization as well
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X)
    #inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)

#Plot

plt.figure(figsize=(10,5))
sns.lineplot(x=range(1, 30), y=wcss, marker='o', color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters') 
plt.ylabel('wcss')
plt.show()      

min_scal=MinMaxScaler()
X_scaled=min_scal.fit_transform(X)
X_scaled=pd.DataFrame(X_scaled,columns=['Age','Annual Income (k$)'])

#fitting k-means  to the dataset
kmeans_scaled=KMeans(n_clusters=5,init='k-means++',random_state=42)
#init: number of time the k-means algorithm will be with different centorid seeds.
#the final results will be the best output of n_init consecutive runs in term of inertia
y_kmeans_scaled=kmeans_scaled.fit_predict(X_scaled)

#visualising the clusters
plt.figure(figsize=(15, 7))

# Use DataFrame directly and specify hue for different clusters
sns.scatterplot(data=X_scaled, x='Age', y='Annual Income (k$)', hue=y_kmeans_scaled, palette='viridis', s=50)

# Plot centroids
sns.scatterplot(x=kmeans_scaled.cluster_centers_[:, 0], y=kmeans_scaled.cluster_centers_[:, 1], color='red', label='centroids', s=300, marker='o')

plt.grid(True)
plt.title('Cluster of customers')
plt.xlabel('Age')
plt.ylabel('Annual income')
plt.legend()
plt.show()
