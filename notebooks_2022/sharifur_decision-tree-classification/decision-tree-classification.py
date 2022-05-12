#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn import tree
import numpy as np


# In[ ]:


df = pd.read_csv('../input/decision-tree-classification/shop data.csv')


# In[ ]:


df.head()


# In[ ]:


x = df.iloc[:,:-1]


# In[ ]:


x


# In[ ]:


y=df.iloc[:,4:]


# In[ ]:


y


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


le_x=LabelEncoder()


# In[ ]:


x=x.apply(LabelEncoder().fit_transform)


# In[ ]:


x


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
import numpy as np


# In[ ]:


dtf = DecisionTreeClassifier()


# In[ ]:


dtf.fit(x.iloc[:,0:4],y)


# In[ ]:


xinput = np.array([1,1,0,0])


# In[ ]:


xinput


# In[ ]:


y_predict = dtf.predict([xinput])


# In[ ]:


y_predict


# In[ ]:




