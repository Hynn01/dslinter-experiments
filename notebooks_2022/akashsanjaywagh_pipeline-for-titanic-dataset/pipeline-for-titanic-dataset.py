#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import SelectKBest,chi2

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#importing dataset
df = pd.read_csv("../input/titanic-modify-dataset/train.csv")
df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


# there are 177 missing values in age and 2 missing values in embarked
df.isnull().sum()


# In[ ]:


df.duplicated().sum()


# In[ ]:


plt.figure(figsize=(15,7))
sns.heatmap(df.corr(),annot=True,cmap="coolwarm")


# In[ ]:


df.head(2)


# In[ ]:


# dropping unnessary columns
df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)


# In[ ]:


# Step 1 -> splitting dataset into train and test train/test/split
X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['Survived']),
                                                 df['Survived'],
                                                 test_size=0.2,
                                                random_state=42)


# In[ ]:


X_train.sample(4)


# In[ ]:


y_train.sample(4)


# In[ ]:


#applying column trnsformer to age and embarked column   - imputation transformer
trf1 = ColumnTransformer([
    ('impute_age',SimpleImputer(),[2]),
    ('impute_embarked',SimpleImputer(strategy='most_frequent'),[6])
],remainder='passthrough')


# In[ ]:


# Applying one hot encoding to sex and embarked column - one hot encoding
trf2 = ColumnTransformer([
    ('ohe_sex_embarked',OneHotEncoder(sparse=False,handle_unknown='ignore'),[1,6])
],remainder='passthrough')


# In[ ]:


# Applying min-max Scaling to all data
trf3 = ColumnTransformer([
    ('scale',MinMaxScaler(),slice(0,10))
])


# In[ ]:


# Will be selecting top 8 features - Feature selection
trf4 = SelectKBest(score_func=chi2,k=8)


# In[ ]:


# train the model
trf5 = DecisionTreeClassifier()


# In[ ]:


pipe = Pipeline([
    ('trf1',trf1),
    ('trf2',trf2),
    ('trf3',trf3),
    ('trf4',trf4),
    ('trf5',trf5)
])


# In[ ]:


# Display Pipeline

from sklearn import set_config
set_config(display='diagram')


# In[ ]:


# train
pipe.fit(X_train,y_train)


# In[ ]:


# Predict
y_pred = pipe.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# ## Cross Validation using Pipeline

# In[ ]:


from sklearn.model_selection import cross_val_score
cross_val_score(pipe,X_train,y_train,cv=10,scoring="accuracy").mean()


# ## Applying gridsearch to pipeline

# In[ ]:


# gridsearchcv
params = {
    'trf5__max_depth':[1,2,3,4,5,None]
}


# In[ ]:


from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)


# In[ ]:


grid.best_score_


# In[ ]:


grid.best_params_


# ## Exporting the pipeline

# In[ ]:


# export 
import pickle
pickle.dump(pipe,open('pipe.pkl','wb'))


# In[ ]:




