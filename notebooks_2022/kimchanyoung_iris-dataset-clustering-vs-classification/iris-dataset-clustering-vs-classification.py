#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <div class="text_cell_render border-box-sizing rendered_html">
# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:purple;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# [❃ Iris Dataset ❃] Clustering VS Classification
# </h1>
# </div>
# </div>

# <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile2.uf.tistory.com%2Fimage%2F9909CC3E5E5F11963802DB">

# In[ ]:


pd_data = pd.read_csv('../input/iris/Iris.csv')
x_train = pd_data.iloc[:,1:-1]
y_train = pd_data.iloc[:,-1]


# In[ ]:


count_list = [(pd_data.Species == 'Iris-setosa').sum(), (pd_data.Species == 'Iris-versicolor').sum(), (pd_data.Species == 'Iris-virginica').sum()]
label_list = list(pd_data['Species'].unique())
plt.figure(figsize = (10, 7))
plt.pie(count_list, labels = label_list, autopct = "%.2f %%", startangle = 90, explode = (0.1, 0.1, 0.0), textprops = {'fontsize': 12})
plt.title('Distribution of Classes', fontsize = 20)
plt.show()


# In[ ]:


data = pd_data.iloc[:,1:]
sns.pairplot(data, hue = 'Species')
plt.figure(figsize = (5,5))
plt.show()


# In[ ]:


corr = data.corr()
fig,axis = plt.subplots(figsize = (6,4))
sns.heatmap(corr,annot = True, ax = axis,linewidths=.5,cmap="Blues")


# In[ ]:


data.groupby(by = "Species").mean()
data.groupby(by = "Species").mean().plot(kind="bar")
plt.title('Class vs Measurements')
plt.ylabel('mean measurement(cm)')
plt.xticks(rotation=0) 
plt.grid(True)
plt.legend(loc="upper left", bbox_to_anchor=(1,1))


# <div class="text_cell_render border-box-sizing rendered_html">
# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:purple;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# Visualizing 3D t-SNE
# </h1>
# </div>
# </div>

# In[ ]:


from sklearn.manifold import TSNE
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


tsne = TSNE(random_state = 42, n_components=3, verbose=0, perplexity=40, n_iter=1000).fit_transform(x_train)

y_train_label = y_train.astype(str)
fig = px.scatter_3d(x_train, x=tsne[:,0],
                 y=tsne[:,1],
                 z=tsne[:,2],
                 color=y_train_label)
fig.update_layout(
                margin=dict(l=20, r=20, t=20, b=20)
)
fig.update_traces(marker_size=5)
iplot(fig)


# In[ ]:


def visualization(y, algo, title):
    x = x_train
    plt.scatter(x[y == 0, 0], x[y == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
    plt.scatter(x[y == 1, 0], x[y == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
    plt.scatter(x[y == 2, 0], x[y == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')
    
    if algo == kmeans:
        plt.scatter(algo.cluster_centers_[:, 0], algo.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
    plt.title(title)
    plt.legend()


# <div class="text_cell_render border-box-sizing rendered_html">
# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:purple;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# Data Preprocessing
# </h1>
# </div>
# </div>

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)


# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)


# <div class="text_cell_render border-box-sizing rendered_html">
# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:purple;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# Kmeans Clustering
# </h1>
# </div>
# </div>

# In[ ]:


from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, 
                    init = 'k-means++', 
                    max_iter = 300, 
                    n_init = 10, 
                    random_state = 0)
    kmeans.fit(x_train)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss,marker = 'o')
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters = 3, 
                init = 'k-means++', 
                max_iter = 300, 
                n_init = 10, 
                random_state = 0)
y_kmeans = kmeans.fit_predict(x_train)


# In[ ]:


from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import silhouette_score

kmeans_score = adjusted_rand_score(y_train, y_kmeans)
kmeans_silhouette = silhouette_score(x_train, y_kmeans, metric='euclidean', sample_size=None, random_state=None)
visualization(y_kmeans, kmeans, "Kmeans AR: {:.2f}, SC: {:.2f}".format(kmeans_score, kmeans_silhouette))


# <div class="text_cell_render border-box-sizing rendered_html">
# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:purple;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# DBSCAN
# </h1>
# </div>
# </div>

# In[ ]:


from sklearn.neighbors import NearestNeighbors

neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(x_train)
distances, indices = nbrs.kneighbors(x_train)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)


# In[ ]:


from sklearn.cluster import DBSCAN

db = DBSCAN(eps = 0.6)
y_dbscan = db.fit_predict(x_train)


# In[ ]:


db_score = adjusted_rand_score(y_train, y_dbscan)
db_silhouette = silhouette_score(x_train, y_dbscan, metric='euclidean', sample_size=None, random_state=None)
visualization(y_dbscan, db, "DBSCAN AR: {:.2f}, SC: {:.2f}".format(db_score,db_silhouette))


# <div class="text_cell_render border-box-sizing rendered_html">
# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:purple;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# XGBoost
# </h1>
# </div>
# </div>

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=.2, random_state=0, stratify = y_train)


# In[ ]:


import xgboost as xgb
from sklearn.metrics import accuracy_score

xgb_clf = xgb.XGBClassifier(random_state=0)
xgb_clf = xgb_clf.fit(X_train, Y_train)

Y_xgb = xgb_clf.predict(X_test)


# In[ ]:


print("XGBoost Accuracy : {:.3f}".format(accuracy_score(Y_test, Y_xgb)))


# <div class="text_cell_render border-box-sizing rendered_html">
# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:purple;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# Decision Tree
# </h1>
# </div>
# </div>

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

