#!/usr/bin/env python
# coding: utf-8

# # **Using XGBoost to classify the results of the 'Acoustic Extinguisher Fire Dataset'**

# ## **Let's import the core libraries**

# In[ ]:


get_ipython().system(' pip install openpyxl')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
from scipy.stats import skew

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


# ## **Now read data from excel spreadsheet**

# In[ ]:


df = pd.read_excel('../input/acoustic-extinguisher-fire-dataset/Acoustic_Extinguisher_Fire_Dataset/Acoustic_Extinguisher_Fire_Dataset.xlsx')
df.head()


# ## **Collect some information about our data**

# In[ ]:


df.info()


# In[ ]:


df.isna().sum()


# In[ ]:


df.describe()


# In[ ]:


oe = OrdinalEncoder()
df['FUEL'] = oe.fit_transform(df[['FUEL']])


# In[ ]:


oe.categories_


# In[ ]:


df.head()


# ## **For a better understanding, let's build some graphs**

# ### **Let's look at the distribution of fuels on a pie chart**

# In[ ]:


df['FUEL'].value_counts().plot(kind='pie', autopct='%.2f%%')
plt.show()


# ### **Let's look at the skewness of our data**

# In[ ]:


for col in df:
    print(f'Col name: {col}')
    print(f'Skewness: {skew(df[col])}')
    
    plt.figure(figsize=(10,8))
    sns.distplot(df[col])
    plt.grid(True)
    plt.show()


# ### **For a better understanding of the data correlation, we will build a heat map**

# In[ ]:


plt.figure(figsize=(10,5))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.show()


# ### **Let's take a look at the uniqueness of values**

# In[ ]:


plt.figure(figsize=(10,5))
plt.bar(df.columns, df.nunique())
plt.show()


# ## **Preparing data for classification**

# In[ ]:


df.columns


# In[ ]:


x = df.iloc[:,:-1]
x.head()


# In[ ]:


y = df.iloc[:, -1]
y.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# In[ ]:


st_sc = StandardScaler()
X_train = st_sc.fit_transform(X_train)
X_test = st_sc.fit_transform(X_test)


# ## **Training and results of the XGBoost model**

# In[ ]:


xg = XGBClassifier()
xg.fit(X_train, y_train)
y_pred = xg.predict(X_test)
print(classification_report(y_test, y_pred))


# In[ ]:


print(f'Model actual accuracy: {accuracy_score(y_test, y_pred)}')


# In[ ]:


confusion_matrix(y_test, y_pred)

