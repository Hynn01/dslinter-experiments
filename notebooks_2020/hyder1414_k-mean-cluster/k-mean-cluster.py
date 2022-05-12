#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#%%


import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('../input/Clustering3')
df.head()


# In[ ]:


plt.scatter(df['Cost'], df['Sales'])


# In[ ]:


km = KMeans(n_clusters=3)
km


# In[ ]:


y_predicted = km.fit_predict(df)
y_predicted


# In[ ]:


df['cluster'] = y_predicted
df.head()


# In[ ]:


df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

plt.scatter(df1.Cost,df1['Sales'],color='red')
plt.scatter(df2.Cost,df2['Sales'], color='blue')
plt.scatter(df3.Cost,df3['Sales'], color='black')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Cost')
plt.ylabel('Sale')
plt.legend()


# In[ ]:


# Normalization
scaler=MinMaxScaler()
scaler.fit(df[['Cost']])
df['Cost'] = scaler.transform(df[['Cost']])

scaler.fit(df[['Sales']])
df['Sales']= scaler.transform(df[['Sales']])
df.head()


# In[ ]:


#train data set
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Cost', 'Sales']])
y_predicted


# In[ ]:


df['cluster']=y_predicted
df.head()


# In[ ]:


km.cluster_centers_


# In[ ]:


df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

plt.scatter(df1.Cost,df1['Sales'],color='red')
plt.scatter(df2.Cost,df2['Sales'], color='blue')
plt.scatter(df3.Cost,df3['Sales'], color='black')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Cost')
plt.ylabel('Sale')
plt.legend()


# In[ ]:


sse = []

k_range = range(1,10)

for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(df[['Cost','Sales']])
    sse.append(km.inertia_)
    


# In[ ]:


plt.xlabel('K')
plt.ylabel('Square error sum')
plt.plot(k_range,sse)


# ![](http://)
