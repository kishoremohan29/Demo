#!/usr/bin/env python
# coding: utf-8

# # Customer Clustering with K-Means
# You work for an e-commerce company, and your task is to group customers into distinct clusters based 
# on their shopping behavior and preferences using the K-Means clustering algorithm. The dataset contains 
# customer information, purchase history, and browsing data. Your goal is to create customer clusters for 
# targeted marketing and personalized recommendations. Answer the following questions based on this 
# case study:
# 1. Data Exploration:
# a. Load the customer dataset using Python libraries like pandas and explore its structure. Describe 
# the features and the data distribution.
# b. Discuss the importance of customer clustering in the e-commerce industry.
# 2. Data Preprocessing:
# a. Prepare the customer data for clustering. Discuss the steps involved in data preprocessing, such 
# as scaling, handling missing values, and encoding categorical variables.
# 3. Implementing K-Means:
# a. Implement the K-Means clustering algorithm using Python libraries like scikit-learn to cluster 
# customers based on their features.
# b. Choose an appropriate number of clusters (K) for the algorithm and explain your choice.
# 4. Model Training:
# a. Train the K-Means model using the preprocessed customer dataset.
# b. Discuss the distance metric used for cluster assignment and its significance in customer 
# clustering.
# 5. Customer Clustering:
# a. Assign customers to their respective clusters based on their features.
# b. Visualize the customer clusters and analyze the characteristics of each cluster.
# 6. Performance Metrics:
# a. Explain the concept of silhouette score and how it is used to evaluate the quality of clustering.
# b. Calculate the silhouette score for the customer clusters and interpret the results.
# 7. Hyperparameter Tuning:
# a. Describe the impact of the number of clusters (K) on the performance of K-Means and suggest 
# strategies for selecting the optimal value of K.
# b. Conduct hyperparameter tuning for the K-Means model and discuss the impact of different 
# values of K on clustering results.
# 8. Real-World Application:
# a. Describe the practical applications of customer clustering in the e-commerce industry.
# b. Discuss how customer clustering can lead to improved customer engagement, targeted 
# marketing, and personalized recommendations.
# 9. Model Limitations:
# a. Identify potential limitations of the K-Means clustering algorithm in customer segmentation 
# and discuss scenarios in which it may not perform well.
# 10. Presentation and Recommendations:
# a. Prepare a presentation or report summarizing your analysis, results, and recommendations for 
# the e-commerce company. Highlight the significance of customer clustering and the role of KMeans in personalized marketing.
# In this case study, you are required to demonstrate your ability to use the K-Means clustering algorithm 
# for customer segmentation, understand the importance of performance metrics like silhouette score, and 
# communicate the practical applications of customer clustering in the e-commerce sector

# In[1]:


import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(style="darkgrid")

from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler

from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_samples, silhouette_score

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = pd.read_csv("C://Users//kishore//Mall_Customers.csv")


# In[4]:


#Let's see how our data looks like!

data.head(5)


# In[5]:


# Let's get some more information about our dataset.

data.info()


# In[6]:


import missingno as mn
mn.matrix(data)


# # Plotting the data:

# In[7]:


plt.figure(figsize=(8,5))
plt.scatter('Annual Income (k$)','Spending Score (1-100)',data=data, s=30, color="red", alpha = 0.8)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')


# # Rescaling

# In[8]:


x= data.iloc[:,3:5]

x_array =  np.array(x)
print(x_array)


# In[9]:


scaler = StandardScaler() 

x_scaled = scaler.fit_transform(x_array)
x_scaled


# In[10]:


# Fitting the model for values in range(1,11)

SSD =[]
K = range(1,11)

for k in K:
    km = KMeans(n_clusters = k)
    km = km.fit(x_scaled)
    SSD.append(km.inertia_)


# In[11]:


#plotting Elbow
plt.figure(figsize=(8,5))
plt.plot(K, SSD, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal K')
plt.show()


# # 2) Silhouette Coefficient Method:

# In[12]:


KMean= KMeans(n_clusters=5)
KMean.fit(x_scaled)
label=KMean.predict(x_scaled)

print("Silhouette Score(n=5):",silhouette_score(x_scaled, label))


# In[13]:


model = KMeans(random_state=123)

# Instantiate the KElbowVisualizer with the number of clusters and the metric 
Visualizer = KElbowVisualizer(model, k=(2,6), metric='silhouette', timings=False)
plt.figure(figsize=(8,5))
# Fit the data and visualize
Visualizer.fit(x_scaled)    
Visualizer.poof()


# In[14]:


print(KMean.cluster_centers_)


# In[15]:


print(KMean.labels_)


# In[16]:


#Add cluster results columns to the dataset dataframe

data["cluster"] = KMean.labels_
data.head()


# In[17]:


plt.figure(figsize=(8,5))

plt.scatter(x_scaled[label==0, 0], x_scaled[label==0, 1], s=100, c='red', label ='Careless')
plt.scatter(x_scaled[label==1, 0], x_scaled[label==1, 1], s=100, c='blue', label ='Target')
plt.scatter(x_scaled[label==2, 0], x_scaled[label==2, 1], s=100, c='green', label ='Planner')
plt.scatter(x_scaled[label==3, 0], x_scaled[label==3, 1], s=100, c='cyan', label ='Sensible')
plt.scatter(x_scaled[label==4, 0], x_scaled[label==4, 1], s=100, c='magenta', label ='Moderate')

plt.title('Cluster of Clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show


# In[ ]:




