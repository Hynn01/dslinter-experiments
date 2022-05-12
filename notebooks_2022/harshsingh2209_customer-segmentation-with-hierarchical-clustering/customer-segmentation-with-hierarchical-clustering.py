#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Importing the Dataset

# In[ ]:


data = pd.read_csv('../input/customer-clustering/segmentation data.csv')


# In[ ]:


data.shape


# In[ ]:


data.drop(['ID'], inplace=True, axis=1)


# In[ ]:


data.head(10)


# In[ ]:


data.describe()


# In[ ]:


data.isna().sum()


# ## Exploratory Data Analysis

# In[ ]:


plt.figure(figsize=(21,15))

plt.subplot2grid((2,2), (0,0))
box1 = sns.boxplot(y=data.Age)
plt.title("Age")

plt.subplot2grid((2,2), (0,1))
box2 = sns.boxplot(y=data.Income)
plt.title("Income")

plt.show()


# In[ ]:


data.Age.describe()


# In[ ]:


data.Income.describe()


# ### Inferences
# - Mean age is approximately 36 years. Max is 76 meanwhile Min is 18
# - Mean income is 121k. Max is 310k meanwhile Min is 36k

# ### Proportion of data values in each feature

# In[ ]:


plt.figure(figsize=(21,15))

plt.subplot2grid((3,3), (0,0))
sns.histplot(data.Sex.astype(str), stat='proportion')

plt.subplot2grid((3,3), (0,1))
sns.histplot(data['Marital status'].astype(str), stat='proportion')

plt.subplot2grid((3,3), (0,2))
sns.histplot(data.Education.astype(str).sort_values(), stat='proportion')

plt.subplot2grid((3,3), (1,0))
sns.histplot(data.Occupation.astype(str).sort_values(), stat='proportion')

plt.subplot2grid((3,3), (1,1))
sns.histplot(data['Settlement size'].astype(str).sort_values(), stat='proportion')

plt.show()


# ## K Means Model

# In[ ]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[ ]:


wcss = {'wcss_score':[], 'no_of_clusters':[]}
for i in range(1,11):
    kmeans = KMeans(i, random_state=0)
    kmeans.fit(data)
    wcss['wcss_score'].append(kmeans.inertia_)
    wcss['no_of_clusters'].append(i)
wcss_df = pd.DataFrame(wcss)


# In[ ]:


wcss_df.head(10)


# In[ ]:


plt.figure(figsize=(14,10))
plt.plot(wcss_df.no_of_clusters, wcss_df.wcss_score, marker='o')
plt.title("Elbow Method to determine number of clusters(K)")
plt.show()


# ### Inference
# Number of clusters in this dataset are 4
# - K = 4

# In[ ]:


kmeans_final = KMeans(n_clusters=4, random_state=0, init='k-means++')
classlabels = kmeans_final.fit_predict(data)


# In[ ]:


data['classlabels'] = classlabels
data.classlabels = data.classlabels.astype(str)
data = data.sort_values('classlabels')


# In[ ]:


plt.figure(figsize=(14,10))
sns.histplot(data.classlabels)
plt.show()


# In[ ]:


score = silhouette_score(data, kmeans_final.labels_, random_state=0)
print(f"Silhouette score: {score:0.3f} ~ 0")


# Silhouette score of 0 means our model did not work very well. The worse could be -1, but the best can go upto 1.

# ## Hierarchical clustering - Agglomerative

# To improve the clustering model, we move to hierarchical clustering

# In[ ]:


new_data = data.drop(['classlabels'], axis=1)


# In[ ]:


from sklearn.cluster import AgglomerativeClustering


# ### Distances and Linkages
# With multiple computation options for both distance and linkage in clusters, we calculate the silhouette score for all permutations

# In[ ]:


## function to compute scores for all permutations
def s_score(distance, linkage):
    agc = AgglomerativeClustering(n_clusters=4, affinity=distance, linkage=linkage)
    agc.fit_predict(new_data)
    score = silhouette_score(new_data, agc.labels_, random_state=0)
    return score


# In[ ]:


distances = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
linkages = ['ward', 'complete', 'average', 'single']


# In[ ]:


scoring = {'dist':[], 'link':[], 'sScore':[]}
for i in distances:
    for j in linkages:
        try:
            score = s_score(i, j)
            scoring['dist'].append(i)
            scoring['link'].append(j)
            scoring['sScore'].append(score)
        except:
            scoring['dist'].append(i)
            scoring['link'].append(j)
            scoring['sScore'].append(np.nan)
scoringDf = pd.DataFrame(scoring)


# We put this process in try-except block since 'ward' only works with 'euclidean' distance. We can now find the best permutation.

# In[ ]:


scoringDf.dropna(axis=0, inplace=True)


# In[ ]:


scoringDf.head(20)


# In[ ]:


final_result = scoringDf[scoringDf['sScore'] == max(scoringDf['sScore'])]
final_result


# ## Finally
# - ‘single’ uses the minimum of the distances between all observations of the sets. This linkage produces the best result with all distance methods.
# - We produce a silhouette score of 0.704, which is a decent score.
# - This dataset containing information about 2000 customers has been classified into 4 clusters or segments.
