#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import math as ma
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv('../input/iris/Iris.csv')


# ## Here, I have to predicted the species of a flower by training our model on the Iris Dataset with the given input points using Naive Bayes.
# 
# ## Input Points
# 
# ## {SL=4.7, SW=3.7,PL=2,PW=0.3}

# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# ## P(Species/SL=4.7, SW=3.7,PL=2,PW=0.3)=P(SL=4.7/Species)P(SW=3.7/Species)P(PL=2/Species)P(PW=0.3/Species)P(Species)

# In[ ]:


plt.subplot(221)
sns.distplot(df['SepalLengthCm'])
plt.subplot(222)
sns.distplot(df['SepalWidthCm'])
plt.subplot(223)
sns.distplot(df['PetalLengthCm'])

plt.subplot(224)
sns.distplot(df['PetalWidthCm'])
plt.show()


# In[ ]:


df['Species'].value_counts()


# In[ ]:


P=50/100


# ## SD Determination

# In[ ]:


df[df['Species']=='Iris-setosa'].std() 


# In[ ]:


df[df['Species']=='Iris-versicolor'].std() 


# In[ ]:


df[df['Species']=='Iris-virginica'].std() 


# ## Mean Determination

# In[ ]:


df[df['Species']=='Iris-setosa'].mean() 


# In[ ]:


df[df['Species']=='Iris-versicolor'].mean() 


# In[ ]:


df[df['Species']=='Iris-virginica'].mean() 


# In[ ]:


def cal(x,m,s):
    num = ma.exp(-0.5*np.square((x-m)/s))
    den = s*ma.sqrt(2*ma.pi)
                              
    return(num/den)


# In[ ]:


## Iris-setosa
                   #std(s)   #mean(m)
#SepalLengthCm     0.352490   5.006
#SepalWidthCm      0.381024   3.418
#PetalLengthCm     0.173511   1.464
#PetalWidthCm      0.107210   0.244


# In[ ]:


P_Setosa=cal(4.7,5.006,0.352490)*cal(3.7,3.418,0.381024)*cal(2,1.464,1.173511)*cal(0.3,0.244,0.107210)*P
P_Setosa


# In[ ]:


## Iris-versicolor
                   #std(s)   #mean(m)
#SepalLengthCm     0.516171   5.936     
#SepalWidthCm      0.313798   2.770
#PetalLengthCm     0.469911   4.260
#PetalWidthCm      0.197753   1.326


# In[ ]:


P_Versicolor=cal(4.7,5.936,0.516171)*cal(3.7,2.770,0.313798)*cal(2,4.260,0.469911)*cal(0.3,1.326, 0.197753)*P
P_Versicolor


# In[ ]:


## Iris-virginica
                  #std(s)    #mean(m)
#SepalLengthCm     0.635880   6.588
#SepalWidthCm      0.322497   2.974
#PetalLengthCm     0.551895   5.552
#PetalWidthCm      0.274650   2.026


# In[ ]:


P_Virginica=cal(4.7,6.588,0.635880)*cal(3.7,2.974,0.322497)*cal(2,5.552,0.551895)*cal(0.3,2.026,0.274650)*P
P_Virginica


# In[ ]:


P_Setosa>P_Virginica


# In[ ]:


P_Setosa>P_Versicolor


# ## So, according to Naive Bayes the flower is Setosa

# ## Let's see the model's prediction

# In[ ]:


probability,species = [P_Setosa,P_Versicolor,P_Virginica],['Iris-setosa','Iris-versicolor','Iris-virginica']
print('Predicted species is : {}'.format(species[probability.index(max(probability))]))


# In[ ]:




