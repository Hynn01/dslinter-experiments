#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## **Load Data**

# In[ ]:


iris_df = pd.read_csv('/kaggle/input/iris/Iris.csv')
iris_df.head()


# ## **EDA**

# In[ ]:


iris_df.drop('Id', axis=1, inplace=True)
sns.pairplot(iris_df, hue='Species')


# In[ ]:


iris_df['Species'].value_counts().plot(kind='bar')


# ## **Split Data for Clustering**

# In[ ]:


iris_df['Species'].replace('Iris-setosa', 0, inplace=True)
iris_df['Species'].replace('Iris-versicolor', 1, inplace=True)
iris_df['Species'].replace('Iris-virginica', 2, inplace=True)


# In[ ]:


iris_features = iris_df.drop(['Species'], axis=1)
iris_features.head()


# In[ ]:


iris_target = iris_df['Species']
iris_target.value_counts()


# ## **K-Means**

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
scaler.fit_transform(iris_features)

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300).fit(iris_features)


# In[ ]:


iris_df['k-means'] = kmeans.labels_
iris_df.head()


# In[ ]:


pd.crosstab(iris_df['Species'], iris_df['k-means'])


# In[ ]:


# Cluster Evaluation by silhouette score

from sklearn.metrics import silhouette_samples, silhouette_score

iris_df['silhouette_kmeans'] = silhouette_samples(iris_features, iris_df['k-means'])
average_score = silhouette_score(iris_features, iris_df['k-means'])

print('K-means Silhouette Analysis Score :', average_score)
iris_df.head()


# In[ ]:


iris_df.groupby('k-means')['silhouette_kmeans'].mean()


# In[ ]:


# K-Means Result

plt.figure(figsize=(20, 5))

plt.suptitle('K-Means', fontsize=20)

plt.subplot(1, 3, 1)
sns.scatterplot(x=iris_df.iloc[:,0], y=iris_df.iloc[:,1], hue=iris_df['k-means'])

plt.subplot(1, 3, 2)
sns.scatterplot(x=iris_df.iloc[:,0], y=iris_df.iloc[:,2], hue=iris_df['k-means'])

plt.subplot(1, 3, 3)
sns.scatterplot(x=iris_df.iloc[:,0], y=iris_df.iloc[:,3], hue=iris_df['k-means'])


# ## **Mean Shift**

# In[ ]:


from sklearn.cluster import MeanShift, estimate_bandwidth

# best bandwidth for mean shift
best_bandwidth = estimate_bandwidth(iris_features)
print('best_bandwidth :', best_bandwidth)

meanshift = MeanShift(bandwidth=best_bandwidth).fit_predict(iris_features)
print('cluster labels :', np.unique(meanshift))


# In[ ]:


iris_df['meanshift'] = meanshift
iris_df.head()


# In[ ]:


# Mean Shift Result

plt.figure(figsize=(20, 5))

plt.suptitle('Mean Shift', fontsize=20)

plt.subplot(1, 3, 1)
sns.scatterplot(x=iris_df.iloc[:,0], y=iris_df.iloc[:,1], hue=iris_df['meanshift'])

plt.subplot(1, 3, 2)
sns.scatterplot(x=iris_df.iloc[:,0], y=iris_df.iloc[:,2], hue=iris_df['meanshift'])

plt.subplot(1, 3, 3)
sns.scatterplot(x=iris_df.iloc[:,0], y=iris_df.iloc[:,3], hue=iris_df['meanshift'])


# ## **GMM**

# In[ ]:


from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3).fit_predict(iris_features)

iris_df['gmm'] = gmm
iris_df.head()


# In[ ]:


pd.crosstab(iris_df['Species'], iris_df['gmm'])


# In[ ]:


# Cluster Evaluation by silhouette score

iris_df['silhouette_gmm'] = silhouette_samples(iris_features, iris_df['gmm'])
average_score = silhouette_score(iris_features, iris_df['gmm'])

print('GMM Silhouette Analysis Score :', average_score)
iris_df.head()


# In[ ]:


iris_df.groupby('gmm')['silhouette_gmm'].mean()


# In[ ]:


# GMM Result

plt.figure(figsize=(20, 5))

plt.suptitle('GMM', fontsize=20)

plt.subplot(1, 3, 1)
sns.scatterplot(x=iris_df.iloc[:,0], y=iris_df.iloc[:,1], hue=iris_df['gmm'])

plt.subplot(1, 3, 2)
sns.scatterplot(x=iris_df.iloc[:,0], y=iris_df.iloc[:,2], hue=iris_df['gmm'])

plt.subplot(1, 3, 3)
sns.scatterplot(x=iris_df.iloc[:,0], y=iris_df.iloc[:,3], hue=iris_df['gmm'])


# ## **DBSCAN**

# In[ ]:


from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.6, min_samples=8, metric='euclidean').fit_predict(iris_features)

iris_df['dbscan'] = dbscan
iris_df.head()


# In[ ]:


pd.crosstab(iris_df['Species'], iris_df['dbscan'])


# In[ ]:


# Cluster Evaluation by silhouette score

iris_df['silhouette_dbscan'] = silhouette_samples(iris_features, iris_df['dbscan'])
average_score = silhouette_score(iris_features, iris_df['dbscan'])

print('DBSCAN Silhouette Analysis Score :', average_score)
iris_df.head()


# In[ ]:


iris_df.groupby('dbscan')['silhouette_dbscan'].mean()


# In[ ]:


# DBSCAN Result

plt.figure(figsize=(20, 5))

plt.suptitle('DBSCAN', fontsize=20)

plt.subplot(1, 3, 1)
sns.scatterplot(x=iris_df.iloc[:,0], y=iris_df.iloc[:,1], hue=iris_df['dbscan'])

plt.subplot(1, 3, 2)
sns.scatterplot(x=iris_df.iloc[:,0], y=iris_df.iloc[:,2], hue=iris_df['dbscan'])

plt.subplot(1, 3, 3)
sns.scatterplot(x=iris_df.iloc[:,0], y=iris_df.iloc[:,3], hue=iris_df['dbscan'])

