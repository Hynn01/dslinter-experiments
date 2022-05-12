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


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#D0F0C0;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="text-align:center;" 
#          "color:white;">
# <span style="font-size:30px;"> 
# <b> Tabular May </b>
# </div>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for plotting
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# import data
train_data = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv") 
test_data = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")


# ### 1. Train data

# In[ ]:


print(train_data.shape)
print("")
print(train_data.info())


# In[ ]:


# train data 
train_data.describe().T.style.bar(subset=['mean'], color ='#205ff2')                            .background_gradient(subset=['std'], cmap='coolwarm')                            .background_gradient(subset=['50%'], cmap='coolwarm')


# In[ ]:


train_data["f_27"]


# In[ ]:


train_data["f_27"].value_counts()


# In[ ]:


print(train_data["id"].nunique())
print(train_data["id"].max())
print(train_data["id"].min())


# ### 2. Test data

# In[ ]:


print(test_data.shape)
print(test_data.info())


# In[ ]:


# test data 
test_data.describe().T.style.bar(subset=['mean'], color ='#205ff2')                            .background_gradient(subset=['std'], cmap='coolwarm')                            .background_gradient(subset=['50%'], cmap='coolwarm')


# In[ ]:





# In[ ]:





# In[ ]:





# ### 3. Target

# In[ ]:


train_data["target"]


# In[ ]:


print(train_data.target.value_counts())
print("")
sns.countplot(x=train_data['target'], palette ='Paired')
plt.show()


# In[ ]:


plt.figure(figsize=(10,8))
#plt.title('target')
sns.distplot(train_data['target'], color='forestgreen')
plt.ylabel('Density', fontsize=13)
plt.xlabel('target', fontsize=13)

plt.show()


# In[ ]:


feature_cols = [col for col in train_data.columns if "f_" in col]
dtype_cols = [train_data[i].dtype for i in feature_cols]

dtypes = pd.DataFrame({"features":feature_cols, 
                       "dtype":dtype_cols})

float_cols = dtypes.loc[dtypes["dtype"] == "float64", "features"].values.tolist()
int_cols = dtypes.loc[dtypes["dtype"] == "int64", "features"].values.tolist()


plt.subplots(figsize = (15,15))
sns.heatmap(train_data.corr(),
            annot = True, 
            cmap ="coolwarm", 
            fmt = '0.2f', 
            vmin = -1, 
            vmax = 1, 
            cbar = False);


# In[ ]:


plt.subplots(figsize=(20,15))
for i, column in enumerate(float_cols):
    plt.subplot(4,4,i+1)
    plt.hist(train_data[column], bins = 100, color= 'coral')
    plt.title(column)


# In[ ]:


plt.subplots(figsize=(20,15))
for i, column in enumerate(int_cols):
    plt.subplot(4,4,i+1)
    plt.hist(train_data[column], bins = 100, color = 'cornflowerblue')
    plt.title(column)


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#D0F0C0;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="text-align:center;" 
#          "color:white;">
# <span style="font-size:30px;"> 
# <b> Data preprocessing </b>
# </div>

# In[ ]:


# Concat train and test
all_data = pd.concat([train_data, test_data], ignore_index=True)


# In[ ]:


#f_27 컬럼 분할
temp_all = all_data
for i in range(10):
    temp = []
    for j in range(len(all_data)):
        temp.append(all_data['f_27'][j][i])
    temp_all['f_27_' + str(i + 1)] = temp
temp_all


# In[ ]:


#LabelEncoder
#범주형 변수를 수치형 변수로 변경
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
le.fit(labels)
for i in range(10):
    all_data['f_27_' + str(i + 1)] = le.transform(temp_all['f_27_' + str(i + 1)])

all_data


# In[ ]:


train_df = all_data.iloc[train_data.index[0]:train_data.index[-1] + 1].drop(columns = ["f_27"])
test_df = all_data.iloc[train_data.index[-1] + 1:].drop(columns = ["f_27", "target"])


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


test_df.columns


# In[ ]:





# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#D0F0C0;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="text-align:center;" 
#          "color:white;">
# <span style="font-size:30px;"> 
# <b> Data split and Modeling </b>
# </div>

# In[ ]:


X = train_df.drop(columns=['id', 'target'])
y = train_df['target']


# In[ ]:


# 1. train data와 test data를 8:2 비율로 분할
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 42) 

print("X_train:",X_train.shape)
print("y_train:",y_train.shape)
print("X_test:",X_test.shape)
print("y_test:", y_test.shape)


# In[ ]:


# 2. train data에서 순수한 train data와 validation data를 8:2로 다시 분할
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                                  y_train, 
                                                  test_size = 0.2, 
                                                  shuffle = True, 
                                                  random_state = 42) 

print("X_train:",X_train.shape)
print("y_train:",y_train.shape)
print("X_val:",X_val.shape)
print("y_val:", y_val.shape)


# In[ ]:


def report_model(X, y, model):
    y_pred = model.predict(X)
    print("Accuracy: {:.2f}".format(metrics.accuracy_score(y, y_pred) * 100))
    print("roc auc score: {:.2f}".format(metrics.roc_auc_score(y, y_pred)))
    print(metrics.classification_report(y, y_pred))


# ### 1. RF

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble

rfc = ensemble.RandomForestClassifier(n_estimators = 200,
                                      max_depth = 10,
                                      min_samples_leaf = 10,
#                                      min_samples_split = 5,
                                      n_jobs = -1,
                                      random_state = 42
)

rfc.fit(X_train, y_train)


# In[ ]:


from sklearn import metrics

report_model(X_train, y_train, rfc)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn import model_selection

params = {"n_estimators":[50, 100, 200],
          "min_samples_leaf": [5, 10, 20],
#          'min_samples_split' : [5, 10, 20],
          'max_depth' : [5, 10, 15]
}

random_search = model_selection.RandomizedSearchCV(rfc,
                                                   params,
                                                   cv = 2,
                                                   scoring = 'roc_auc',
                                                   verbose = 3,
                                                   refit = True)

random_search.fit(X_train, y_train)
pd.DataFrame(random_search.cv_results_).sort_values('rank_test_score',ascending=True)


# In[ ]:


rfc = ensemble.RandomForestClassifier(n_estimators = 100,
                                      min_samples_leaf = 5,
#                                      min_samples_split = 10,
                                      max_depth = 15,
                                      n_jobs = -1,
                                      random_state = 42)
rfc.fit(X_train, y_train)


# In[ ]:


from sklearn import metrics

report_model(X_train, y_train, rfc)


# ### 2. LightGBM

# In[ ]:


from lightgbm import LGBMClassifier

lgbmc = LGBMClassifier(subsample = 0.8, 
                       n_estimators = 50, 
                       min_data_in_leaf= 20,
                       min_child_samples= 40, 
                       max_depth = 5,          
                       feature_fraction = 1,
                       bagging_fraction = 0.8,
                       n_jobs = -1,
                       random_state = 42
)

lgbmc.fit(X_train, y_train)


# In[ ]:


from sklearn import metrics

report_model(X_train, y_train, lgbmc)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn import model_selection

params = {'max_depth': [2, 5, 10],
          'min_child_samples': [20, 40, 60],
          'subsample': [0.6, 0.8, 1],
          'n_estimators ': [50, 100, 150]
         }

random_search = model_selection.RandomizedSearchCV(lgbmc,
                                                   params,
                                                   cv = 2,
                                                   scoring = 'roc_auc',
                                                   verbose = 3,
                                                   refit = True)

random_search.fit(X_train, y_train)
pd.DataFrame(random_search.cv_results_).sort_values('rank_test_score',ascending=True)


# In[ ]:


lgbmc = LGBMClassifier(subsample = 0.8, 
                                n_estimators = 100, 
                                min_child_samples= 40, 
                                max_depth = 20,          
                                feature_fraction = 1,
                                bagging_fraction = 0.8,
                                n_jobs = -1,
                                random_state = 42
)

lgbmc.fit(X_train, y_train)


# In[ ]:


from sklearn import metrics

report_model(X_train, y_train, lgbmc)


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#D0F0C0;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="text-align:center;" 
#          "color:white;">
# <span style="font-size:30px;"> 
# <b> Test data predicting </b>
# </div>

# In[ ]:


X_test = test_df.drop(columns=['id'])
#y_test = test_df['target']


# In[ ]:


rfc = ensemble.RandomForestClassifier(n_estimators = 100,
                                      min_samples_leaf = 5,
#                                      min_samples_split = 10,
                                      max_depth = 15,
                                      n_jobs = -1,
                                      random_state = 42)
rfc.fit(X_train, y_train)


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#D0F0C0;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="text-align:center;" 
#          "color:white;">
# <span style="font-size:30px;"> 
# <b> Submission </b>
# </div>

# In[ ]:


def test_pred():
    pred_list = []
    for seed in range(5):
        model = ensemble.RandomForestClassifier(n_estimators = 100,
                                      min_samples_leaf = 5,
#                                      min_samples_split = 10,
                                      max_depth = 15,
                                      n_jobs = -1,
                                      random_state = 42) 
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_test)[:,1]
        pred_list.append(preds)
    return pred_list


# In[ ]:


pred_list = test_pred()
pred_df = pd.DataFrame(pred_list).T
pred_df = pred_df.rank()
pred_df["mean"] = pred_df.mean(axis=1)
pred_df


# In[ ]:


sample_sub = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
sample_sub["target"] = pred_df["mean"]
sample_sub


# In[ ]:


sample_sub.to_csv('submission_rfc.csv', index = False)

