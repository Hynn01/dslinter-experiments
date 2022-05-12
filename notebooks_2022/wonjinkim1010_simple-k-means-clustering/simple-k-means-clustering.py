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


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#D0F0C0;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="text-align:center;">
# <span style="font-size:30px;"> 
# <b> K-Means Clustering </b>
# </div>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for plotting
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans


# In[ ]:


iris_data = pd.read_csv("../input/iris/Iris.csv")


# In[ ]:


print(iris_data.shape)
print(iris_data.info())


# In[ ]:


iris_data


# In[ ]:


# ID Column 제거
iris_data = iris_data.drop("Id", axis = 1)


# In[ ]:


# x data, y data split
x = iris_data.iloc[:,0:4]
y = iris_data.iloc[:,-1]


# In[ ]:


x


# In[ ]:


# species -> 범주형을 숫자형으로 변환하지 않고 진행함.
y


# In[ ]:


# species 별로 분포 확인
sns.pairplot(iris_data, 
             hue="Species", 
             diag_kind="hist",
             size=2.0,
            palette="hls");


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#D0F0C0;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="text-align:center;">
# <span style="font-size:25px;"> 
# <b> Data scaling </b>
# </div>

# In[ ]:


# x data scaling : standardscaler 사용

from sklearn import preprocessing

sc = preprocessing.StandardScaler()

sc.fit(x)
x_scaled = sc.transform(x)
x_scaled = pd.DataFrame(x_scaled, 
                        columns = x.columns)

x_scaled.sample(5)


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#D0F0C0;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="text-align:center;">
# <span style="font-size:25px;"> 
# <b> K-means clustering </b>
# </div>

# In[ ]:


from sklearn.cluster import KMeans
# 적절한 군집수 찾기
# Inertia(군집 내 거리제곱합의 합) value (적정 군집수)

ks = range(1,10)
inertias = []

for k in ks:
    model = KMeans(n_clusters = k)
    model.fit(x)
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.figure(figsize=(10,8))

plt.plot(ks, inertias, '-o', color = 'forestgreen')
plt.xlabel('Number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# In[ ]:


x = iris_data.iloc[:, :4].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, 
                    init = 'k-means++', 
                    max_iter = 300, 
                    n_init = 10, 
                    random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
kmeans = KMeans(n_clusters = 3, 
                init = 'k-means++', 
                max_iter = 300, 
                n_init = 10, 
                random_state = 0)

y_kmeans = kmeans.fit_predict(x)


# In[ ]:


y_kmeans


# In[ ]:


print(kmeans.labels_)


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#D0F0C0;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="text-align:center;">
# <span style="font-size:25px;"> 
# <b> Visualization </b>
# </div>

# In[ ]:


plt.figure(figsize=(10,8))


plt.scatter(x[y_kmeans == 0, 0], 
            x[y_kmeans == 0, 1], 
            s = 50, 
            c = 'royalblue', 
            label = 'Iris-setosa')

plt.scatter(x[y_kmeans == 1, 0], 
            x[y_kmeans == 1, 1], 
            s = 50, 
            c = 'forestgreen', 
            label = 'Iris-versicolour')

plt.scatter(x[y_kmeans == 2, 0], 
            x[y_kmeans == 2, 1],
            s = 50, 
            c = 'sandybrown', 
            label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:,1], 
            s = 150, 
            c = 'red', 
            label = 'Centroids')

plt.legend()


# In[ ]:


kmeans = KMeans(n_clusters = 3, 
                init = 'k-means++', 
                max_iter = 300, 
                n_init = 10, 
                random_state = 0)

y_kmeans = kmeans.fit_predict(x)

df = pd.DataFrame({'labels': y_kmeans,
                   "Species": iris_data['Species']})

crosstab = pd.crosstab(df['labels'],
                       df['Species'])


# In[ ]:


plt.figure(figsize=(24,10))
plt.subplot(1,2,1)
plt.title("K-Means Clustering", fontsize=18)
sns.heatmap(crosstab,
            annot = True,
            cbar = False,
            cmap = "bwr")
plt.show()


# In[ ]:




