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


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
train_data.head()


# In[ ]:


# My goal is to increase correlation between data
plt.figure(figsize=(8,6))
sns.heatmap(train_data.corr(),annot=True)


# In[ ]:


# Passenger Id is unique for every raw so ignore it.
# let's start with Pclass

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=train_data,x='Survived',hue='Pclass')
plt.grid(axis='y')


# In[ ]:


# there is good relation between Pclass and Survived


# In[ ]:


# Now Age
# Correlation between Age and Survived is low, Let's increase
print('corr b/w age and survived:',train_data['Age'].corr(train_data['Survived']))
sns.kdeplot(data=train_data,x='Age',hue='Survived')
plt.grid(axis='x')
plt.grid(axis='y')
plt.xticks(np.arange(0,100,10))
plt.show()


# In[ ]:


# There is weak relation between Age and Survived, Let see what we can do.


# In[ ]:


# I am converting continuous variable('Age') to categorical variable by using below function
# I made range by using above kde plot of Age vs Survived
def impute_age(x):
    if x <= 9:
        return 1
    elif x > 9 and  x <= 17:
        return 2
    elif x > 13 and  x <= 37:
        return 3
    elif x > 37 and x <= 75:
        return 4
    else:
        return 5
    
train_data['Age'] = train_data['Age'].apply(impute_age)
print('Improved corr b/w age and survived:',train_data['Age'].corr(train_data['Survived']))


# In[ ]:


sns.kdeplot(data=train_data,x='Fare',hue='Survived')
plt.grid(axis='x')
plt.grid(axis='y')
plt.show()


# In[ ]:


# We improved correlation, Let's do same with 'Fare'
def impute_fare(x):
    if x < 50:
        return 0
    else:
        return 1

print('corr b/w fare and survived:',train_data['Fare'].corr(train_data['Survived']))

train_data['Fare'] = train_data['Fare'].apply(impute_fare)
train_data['Fare'].corr(train_data['Survived'])   

print('Improved corr b/w fare and survived:',train_data['Fare'].corr(train_data['Survived']))


# In[ ]:




