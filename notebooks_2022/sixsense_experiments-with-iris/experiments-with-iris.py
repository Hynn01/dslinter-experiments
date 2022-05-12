#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


data = pd.read_csv("../input/iris/Iris.csv")


# In[ ]:


# see data
data.keys()


# In[ ]:


# data preprocessing with standard scaler
dataX = data.drop(columns=['Id', 'Species']).to_numpy()
dataY = data['Species'].to_numpy()

dataY = np.unique(dataY, return_inverse=True)[1]
for idx in range(len(dataX[0])):
    mean, std = dataX[:, idx].mean(), dataX[:, idx].std()
    dataX[:, idx] = (dataX[:, idx] - mean) / std


# In[ ]:


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# kmeans
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0)
kmeans.fit(dataX)

print(kmeans.labels_)

# pca
pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(dataX)

# visualization
d_1 = pca_transformed[:,0]
d_2 = pca_transformed[:,1]

marker0_idx = [kmeans.labels_==0]
marker1_idx = [kmeans.labels_==1]
marker2_idx = [kmeans.labels_==2]

plt.scatter(x=d_1[marker0_idx], y=d_2[marker0_idx], marker='o')
plt.scatter(x=d_1[marker1_idx], y=d_2[marker1_idx], marker='s')
plt.scatter(x=d_1[marker2_idx], y=d_2[marker2_idx], marker='^')

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('3 Clusteres Visualization by 2 PCA Components')
plt.show()


# In[ ]:


from sklearn.metrics import silhouette_samples, silhouette_score

# evaluate clustering
score_samples = silhouette_samples(dataX, kmeans.labels_)
average_score = silhouette_score(dataX, kmeans.labels_)
print('iris Silhouette Analysis Score:{0:3f}'.format(average_score))


# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# evaluate accuracy
model_num = 5
leaf = 8192
estimators = 512
model_list = []

for _ in range(model_num):
    forestCla = RandomForestClassifier(n_estimators=estimators, max_leaf_nodes=leaf, n_jobs=-1)
    model_list.append(forestCla)

kf = StratifiedKFold(n_splits=model_num, shuffle=True, random_state=1)
for idx, [train_idx, validation_idx] in enumerate(kf.split(dataX, dataY)):
    model_list[idx].fit(dataX[train_idx], dataY[train_idx])
    predict = model_list[idx].predict(dataX[validation_idx])
    mismatch_list = np.where(dataY[validation_idx]!=predict)
    print("{}/{}".format(len(validation_idx) - len(mismatch_list[0]), len(validation_idx)))


# In[ ]:


# ensemble model with voting
result = model_list[1].predict_proba(dataX)/2 + model_list[3].predict_proba(dataX)/2
print(len(np.where(np.argmax(result, axis=-1) == dataY)[0]))


# In[ ]:


# origin models score
for i in range(model_num):
    print(len(np.where(model_list[i].predict(dataX) == dataY)[0]))

