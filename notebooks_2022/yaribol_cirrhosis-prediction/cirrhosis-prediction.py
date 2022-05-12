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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/kaggle/input/cirrhosis-prediction-dataset/cirrhosis.csv')
df = pd.DataFrame(data)
df.head(5)


# In[ ]:


df.info()


# <a id='2'></a><br>
# # Data cleaning

# In[ ]:


import missingno as msno
msno.matrix(df)


# In[ ]:


df_clean = df.dropna()
df_clean


# <a id='1'></a><br>
# # Convert caterorical value to numeral value

# In[ ]:


colum_cat = ['Status','Drug','Sex','Ascites','Hepatomegaly','Spiders','Edema']
for i in colum_cat:
    print('-------------------------')
    print(df[i].value_counts())
    print('-------------------------')


# In[ ]:


for i in colum_cat:
    print(f'Catagory of {i}')
    catlist = df_clean[i].unique()
    for j, val in enumerate(catlist):
         dftobjfinal = df_clean[i].replace({val:j},inplace=True)
         #print(dftobjfinal)
         print(j,val)
    print('--------------------------------')


# In[ ]:


df_clean


# In[ ]:


sns.countplot(df["Stage"], palette="Set3")
plt.title("Stage ",fontsize=10)
plt.show()


# <a id='2'></a><br>
# # Detect Outliers

# In[ ]:


from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor()
y_pred = clf.fit_predict(df_clean)


# In[ ]:


x_score = clf.negative_outlier_factor_
outlier_score = pd.DataFrame()
outlier_score["score"] = x_score

#threshold
threshold2 = -1.5
filtre2 = outlier_score["score"] < threshold2
outlier_index = outlier_score[filtre2].index.tolist()


# In[ ]:


len(outlier_index)


# <a id='3'></a><br>
# # Feature Selection

# In[ ]:


f,ax = plt.subplots(figsize=(14,10))
sns.heatmap(df_clean.corr(), cmap="PuBu", annot=True, linewidths=0.5, fmt= '.2f',ax=ax)
plt.show()


# <a id='4'></a><br>
# # Dealing with Imbalanced Data

# In[ ]:


from imblearn import under_sampling 
from imblearn import over_sampling
from imblearn.over_sampling import SMOTE
from collections import Counter


# In[ ]:


x = df_clean.drop(['Stage','ID'], axis = 1)
y = df_clean.loc[:,'Stage'].values


# In[ ]:


y.shape


# In[ ]:


sm = SMOTE(k_neighbors = 3)
print('Original dataset shape %s' % Counter(y))
x, y = sm.fit_resample(x, y)
print('Resampled dataset shape %s' % Counter(y))


# In[ ]:


sns.countplot(y, palette='Set3')
plt.title("Stage ",fontsize=10)
plt.show()


# <a id='5'></a><br>
# # Data Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)


# <a id='14'></a><br>
# ## Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
r_forest = RandomForestClassifier(criterion = 'entropy', max_depth = 20, n_estimators = 10000)
r_forest.fit(x_train,y_train)
predicted = r_forest.predict(x_test)
score = r_forest.score(x_test, y_test)
rf_score_ = np.mean(score)

print('Accuracy : %.3f' % (rf_score_))


# <a id='15'></a><br>
# ### Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import ROCAUC
from yellowbrick.style import set_palette


# In[ ]:


r_forest_cm = ConfusionMatrix(r_forest, cmap='GnBu')

r_forest_cm.fit(x_train, y_train)
r_forest_cm.score(x_test, y_test)
r_forest_cm.show()


# <a id='11'></a><br>
# ### Classification Report

# In[ ]:


print(classification_report(y_test, predicted))


# <a id='12'></a><br>
# ### ROC Curve

# In[ ]:


visualizer = ROCAUC(r_forest)

set_palette('bold')

visualizer.fit(x_train, y_train)        # Fit the training data to the visualizer
visualizer.score(x_test, y_test)        # Evaluate the model on the test data
visualizer.show()   


# <a id='18'></a><br>
# ### Class Prediction Error

# In[ ]:


visualizer = ClassPredictionError(r_forest)

set_palette('pastel')

visualizer.fit(x_train, y_train)        
visualizer.score(x_test, y_test)      
visualizer.show()


# 
