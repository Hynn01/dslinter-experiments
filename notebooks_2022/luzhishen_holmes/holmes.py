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


import random 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols,glm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#将数据集读入到pandas数据框中
wine=pd.read_csv('/kaggle/input/bigdata-and-datamining-1st-ex/Dataset.csv',sep=',',header=0)
winetest=pd.read_csv('/kaggle/input/bigdata-and-datamining-1st-ex/Testing.csv',sep=',',header=0)
#列名重命名，用下划线替换空格，使之符合python命名规范
wine.columns=wine.columns.str.replace(' ','_')
print(wine.head())
#显示所有变量的描述性统计量
#这些统计量包括总数、均值、标准差、最小值、第25个百分位数、中位数、第75个百分位数和最大值
print(wine.describe())
#找出唯一值
print(sorted(wine.quality.unique()))
#计算值的频率
print(wine.quality.value_counts())


# In[ ]:


import random
winerandom=wine.sample(frac=1)
winerandom.index = range(0,4160) 
print(winerandom)


# In[ ]:


train=winerandom.drop(columns='ID')
a=train.iloc[:,:1]
b=train.iloc[:,1:2]
c=train.iloc[:,2:3]
d=train.iloc[:,3:4]
e=train.iloc[:,4:5]
f=train.iloc[:,5:6]
g=train.iloc[:,6:7]
h=train.iloc[:,7:8]
i=train.iloc[:,8:9]
j=train.iloc[:,9:10]
k=train.iloc[:,10:11]
trainquality=train.iloc[:,11:12]
print(a)


# In[ ]:


import matplotlib.pyplot as plt 
x=np.c_[a,b,c,d,e,f,g,h,i,j,k]
y=train.iloc[:,11:12]
print(x)


# In[ ]:


from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(p=1,n_neighbors=7) 
clf.fit(x,y)


# In[ ]:


score=clf.score(x,y)
print(score)


# In[ ]:


test=winetest.drop(columns='ID')
a=test.iloc[:,:1]
b=test.iloc[:,1:2]
c=test.iloc[:,2:3]
d=test.iloc[:,3:4]
e=test.iloc[:,4:5]
f=test.iloc[:,5:6]
g=test.iloc[:,6:7]
h=test.iloc[:,7:8]
i=test.iloc[:,8:9]
j=test.iloc[:,9:10]
k=test.iloc[:,10:11]


# In[ ]:


test=np.c_[a,b,c,d,e,f,g,h,i,j,k]
y_predicted=clf.predict(test)
y_predicted_rounded=[round(score) for score in y_predicted]
print(y_predicted_rounded)


# In[ ]:


quality= pd.DataFrame(y_predicted_rounded)
quality.columns =['quality']
print(quality)


# In[ ]:


ID=winetest.iloc[:,:1]
ID=pd.DataFrame(ID)
print(ID)


# In[ ]:


working=pd.concat([ID,quality],axis = 1)
print (working)


# In[ ]:


working.to_csv('submission.csv', index=False)


# 这次用的是knn算法
