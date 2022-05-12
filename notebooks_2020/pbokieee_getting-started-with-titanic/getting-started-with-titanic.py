#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # 1、导入数据并进行观察

# In[ ]:


tr_data = pd.read_csv('/kaggle/input/titanic/train.csv')
te_data = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


tr_data.info()


# In[ ]:


tr_data.head()


# In[ ]:


te_data.info()


# In[ ]:


te_data.head()


# # 2、清洗数据

# > 船舱等级、登船港口与生还关联不大，删除Cabin、Embarked

# In[ ]:


tr_data = tr_data.drop(['Cabin','Embarked'],axis = 1)
te_data = te_data.drop(['Cabin','Embarked'],axis = 1)


# > 删除Age中的nan数据

# In[ ]:


tr_data = tr_data.drop(tr_data[tr_data.Age.isna()].index)
#te_data = te_data.drop(te_data[te_data.Age.isna()].index)
te_data.Age.fillna(te_data.Age.mean(),inplace = True)
te_data.Fare.fillna(0,inplace = True)


# >  找出训练集和测试集中object类型的列，object类型且值种类小于10的列，以及数值型的列

# In[ ]:


object_cols = [cname for cname in tr_data.columns if tr_data[cname].dtype == "object"]

low_cardinality_cols  = [cname for cname in tr_data.columns if tr_data[cname].nunique() < 10 and 
                        tr_data[cname].dtype == "object"]
numerical_cols = [cname for cname in tr_data.columns if tr_data[cname].dtype in ['int64', 'float64']]

print(object_cols)
print(low_cardinality_cols)
print(numerical_cols)


# > 删除类型是object并且值种类大于10的列

# In[ ]:


nouse_object_cols = list(set(object_cols)-set(low_cardinality_cols))
tr_data = tr_data.drop(nouse_object_cols,axis = 1)
te_data = te_data.drop(nouse_object_cols,axis = 1)


# > 对Sex进行LabelEncoding

# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_tr_data = tr_data.copy()
label_te_data = te_data.copy()
label_encoder = LabelEncoder()
label_tr_data.Sex = label_encoder.fit_transform(tr_data['Sex'])
label_te_data.Sex = label_encoder.fit_transform(te_data['Sex'])


# In[ ]:


label_tr_data.head()


# In[ ]:


label_te_data.head()


# # 3、建立模型，进行训练和预测

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
y = label_tr_data.Survived
X = label_tr_data.drop('Survived',axis = 1)

model = RandomForestClassifier(criterion = 'entropy',n_estimators=100,n_jobs = 16)
model.fit(X, y)

importances = model.feature_importances_


import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
 
features_list = label_tr_data.columns.values
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
 
plt.figure(figsize=(5,7))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importances')
plt.draw()



predictions = model.predict(label_te_data)

output = pd.DataFrame({'PassengerId': label_te_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)

