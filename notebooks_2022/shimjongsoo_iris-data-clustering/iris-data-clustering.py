#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv("/kaggle/input/iris/Iris.csv")
df


# In[ ]:


df.Species.value_counts()


# In[ ]:


df_cleans = df.drop("Id", axis = 1)
df_cleans


# In[ ]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df_cleans.Species = le.fit_transform(df_cleans.Species)

df_cleans


# ## K-Means
# - Choose K

# In[ ]:


from sklearn.cluster import KMeans

df_kmeans_1 = df_cleans.drop("Species",axis=1)
inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
    algorithm.fit(df_kmeans_1)
    inertia.append(algorithm.inertia_)


# In[ ]:


import matplotlib.pyplot as plt 

plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()


# elbow method => 3 Cluster

# In[ ]:


# Clustering

kmeans = KMeans(n_clusters = 3 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan').fit(df_kmeans_1)
df_kmeans_1['kmeans'] = kmeans.predict(df_kmeans_1)
df_kmeans_1


# In[ ]:


df_kmeans_1.kmeans.value_counts()


# In[ ]:


plt.scatter(df_kmeans_1['SepalLengthCm'], df_kmeans_1['PetalLengthCm'], c = df_kmeans_1.kmeans, alpha = 0.6, 
            linewidths = 0.7, edgecolors = 'red', label = 'spring')
plt.legend()
plt.show()


# In[ ]:


plt.scatter(df_cleans['SepalLengthCm'], df_cleans['PetalLengthCm'], c = df_cleans.Species, alpha = 0.6, 
            linewidths = 0.5, edgecolors = 'blue', label = 'spring')
plt.legend()
plt.show()


# In[ ]:


df_final = df_cleans.copy()
df_final["kmeans"] = df_kmeans_1['kmeans']
df_final


# In[ ]:


df_final.groupby("Species")["kmeans"].value_counts()


# ### include species

# In[ ]:


from sklearn.cluster import KMeans

df_kmeans = df_cleans
inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
    algorithm.fit(df_kmeans)
    inertia.append(algorithm.inertia_)


# In[ ]:


import matplotlib.pyplot as plt 

plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()


# elbow method에 의해 3에서

# In[ ]:


# 클러스터링 진행

kmeans = KMeans(n_clusters = 3 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan').fit(df_kmeans)
df_kmeans['kmeans'] = kmeans.predict(df_kmeans)
df_kmeans


# In[ ]:


df_kmeans.kmeans.value_counts()


# ## DBSCAN
# - Auto Cluster K 

# In[ ]:


from sklearn.cluster import DBSCAN

df_dbscan = df_cleans.drop("Species",axis=1)

dbscan = DBSCAN(eps=0.6, min_samples = 8, metric='euclidean')
dbscan_cluster = dbscan.fit_predict(df_dbscan)

df_final['dbscan'] = dbscan_cluster


# In[ ]:


df_final.dbscan.value_counts()


# In[ ]:


df_final.groupby("Species")["dbscan"].value_counts()


# In[ ]:


plt.scatter(df_final['SepalLengthCm'], df_final['PetalLengthCm'], c = df_final.dbscan, alpha = 0.6, 
            linewidths = 0.7, edgecolors = 'red', label = 'spring')
plt.legend()
plt.show()


# ## GMM
# 

# In[ ]:


from sklearn.mixture import GaussianMixture

df_iris = df_cleans.drop("Species",axis=1)

gmm = GaussianMixture(n_components = 3, random_state = 0)
gmm.fit(df_iris)
gmm_cluster_labels = gmm.predict(df_iris)

df_final['gmm_cluster'] = gmm_cluster_labels

iris_result = df_final.groupby(['Species'])['gmm_cluster'].value_counts()
print(iris_result)


# ## Decision Tree Viz
# 
# Reference
# 
# <a href="https://www.kaggle.com/code/kimchanyoung/iris-dataset-clustering-vs-classification">@CHANYOUNG KIM</a>

# In[ ]:


x_train = df_cleans.iloc[:,1:-1]
y_train = df_cleans.iloc[:,-1]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=.2, random_state=0, stratify = y_train)


# In[ ]:


from sklearn import tree
from sklearn.metrics import plot_confusion_matrix

dt = tree.DecisionTreeClassifier(random_state = 0)
dt.fit(X_train, Y_train)

print("Test Accuracy: ",round(dt.score(X_test, Y_test)*100, 3))

plot_confusion_matrix(dt, X_test, Y_test)

plt.show()


# In[ ]:


get_ipython().system(' pip install dtreeviz')
from dtreeviz.trees import dtreeviz 


# In[ ]:


fn = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
graph = dtreeviz(dt, x_train, y_train,
                 target_name = "Species",
                 feature_names = fn,
                 class_names = list(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']))
graph

