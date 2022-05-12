#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


iris_df = pd.read_csv('../input/iris/Iris.csv')
display(iris_df.head(3))

ft = iris_df.iloc[:,1:-1]
target = iris_df.iloc[:,-1]


# In[ ]:


x = iris_df.Species.unique()
y = iris_df.Species.value_counts()
fig = px.bar(iris_df, 
             x=x, y=y,
             color=x, 
             height=400,
             width=600,
             title='Target Dist.',
             text=y)
fig.update_layout(
    xaxis_title='Species',
    yaxis_title='Counts')
fig.update_xaxes(visible=False)
iplot(fig)


# In[ ]:


index_vals = iris_df.Species.astype('category').cat.codes
fig = go.Figure(data=go.Splom(
                dimensions=[dict(label='Sepal Length',
                                 values=iris_df.SepalLengthCm),
                            dict(label='Sepal Width',
                                 values=iris_df.SepalWidthCm),
                            dict(label='Petal Length',
                                 values=iris_df.PetalLengthCm),
                            dict(label='Petal Width',
                                 values=iris_df.PetalWidthCm)],
                showupperhalf=False,
                text=iris_df.Species,
                marker=dict(color=index_vals, 
                            showscale=False, 
                            line_color='white', 
                            line_width=0.5)
))

fig.update_layout(
    title='Scatter Matrix',
    width=600,
    height=600)

iplot(fig)


# # KMeans

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

sc = StandardScaler()
ft = sc.fit_transform(ft)
le = preprocessing.LabelEncoder()
target = le.fit_transform(target)


# In[ ]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

temp1 = iris_df.iloc[:,1:-1]
kmeans_a = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0)
kmeans_a.fit(temp1)


# In[ ]:


temp1['cluster'] = kmeans_a.labels_
temp1['si_coeff'] = silhouette_samples(ft, temp1['cluster'])
av_score = silhouette_score(ft, temp1['cluster'])
print('Silhouette Score:{0:3f}'.format(av_score))


# In[ ]:


av = temp1.groupby('cluster')['si_coeff'].mean()
cluster = [0,1,2]
avl = list(av.unique())

fig = go.Figure(data=[go.Bar(x=cluster, y=avl,
                     hovertext=['cluster_0', 'cluster_1', 'cluster_2'],
                     text=np.round(avl,3))])

fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)

fig.add_hline(y=0.442, line_dash='dot', 
              annotation_text='average silhouette score:0.442') # average silhouette score

fig.add_hrect(y0=0.335, y1=0.635, line_width=0, fillcolor='red', opacity=0.2)

fig.update_layout(
    title='Silhouette Score',
    xaxis_title='Clusters',
    yaxis_title='Score',
    width=500,
    height=400)

iplot(fig)


# # SVM

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(ft, target, test_size = 0.25, random_state = 0)

sv = SVC(kernel = 'linear', random_state=0)
sv.fit(X_train, y_train)

pred_s = sv.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

acc_score = cross_val_score(estimator=sv, X=X_train, y=y_train, cv=10)
print('Accuracy:{:.2f}%'.format(acc_score.mean()*100))


# In[ ]:


from sklearn.metrics import confusion_matrix, plot_confusion_matrix
labels = iris_df.iloc[:,-1].unique()
plot = plot_confusion_matrix(sv,
                            X_test, y_test,
                            display_labels=labels,
                            cmap= plt.cm.Reds,
                            normalize=None)
plot.ax_.set_title('Confusion Matrix')

