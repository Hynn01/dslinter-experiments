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


import codecs
path = '/kaggle/input/gdp-of-all-countries19602020/gdp_1960_2020.csv'
with codecs.open(path, 'r', 'utf-8', 'ignore') as f:
    gdp = pd.read_csv(f)
gdp[0:2]


# In[ ]:


gdp.columns


# In[ ]:


gdp['country'].unique()


# In[ ]:


india_gdp=gdp[gdp['country']=='India']
india_gdp


# In[ ]:


india_gdp.drop(['rank','state','gdp_percent'], axis = 1, inplace = True)


# In[ ]:


india_gdp


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize = (10,7))
sns.barplot(x = 'year',
y = 'gdp',
hue = 'country',
data = india_gdp)
plt.xticks(rotation = 90)
plt.title("India's GDP")
plt.show()


# In[ ]:


x1 = india_gdp.drop(['gdp', 'country'], axis=1)
y1 = india_gdp['gdp']


# In[ ]:


print(x1.shape)
print(y1.shape)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size = 0.2)


# In[ ]:


model_india=LinearRegression()
model_india.fit(x1,y1)


# In[ ]:


print("Coefficient: ",model_india.coef_)
print("intercept: ",model_india.intercept_)
pred = model_india.predict(x1)


# In[ ]:


plt.figure(figsize=(8,6))
plt.scatter(x1,y1,label='Acutal value')
plt.plot(x1,pred,color='tab:orange',label='Predicted value')
plt.legend()
plt.title("India",color='m')
plt.xlabel("Years",color='r')
plt.ylabel("per year INDIAN gdp",color='c')
plt.tight_layout()
plt.show()


# In[ ]:


years=[2021,2022,2023,2024,2025]
for i in years:
    print(model_india.predict([[i]]))


# In[ ]:


from sklearn.linear_model import LogisticRegression
logimodel_india=LogisticRegression()
logimodel_india.fit(x1,y1)


# In[ ]:


pred_logi = logimodel_india.predict(x1)


# In[ ]:


plt.figure(figsize=(8,6))
plt.scatter(x1,y1,label='Acutal value')
plt.plot(x1,pred_logi,color='tab:orange',label='Predicted value')
plt.legend()
plt.title("India",color='m')
plt.xlabel("Years",color='r')
plt.ylabel("per year INDIAN gdp",color='c')
plt.tight_layout()
plt.show()


# In[ ]:


years=[2020,2021,2022,2023,2024,2025]
for i in years:
    print(logimodel_india.predict([[i]]))

