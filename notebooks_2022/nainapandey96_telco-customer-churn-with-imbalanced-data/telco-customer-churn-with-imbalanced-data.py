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


#First we will do modeling with dataset as it is that is imbalanced data after that we will go through methods to tackle imbalanced data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_churn= pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df_churn.head()


# In[ ]:


df_new=df_churn.drop("customerID", axis=1)
df_new.head()


# In[ ]:


df_new[pd.to_numeric(df_new.TotalCharges, errors="coerce").isnull()]


# In[ ]:


df1= df_new[df_new.TotalCharges !=' ']


# In[ ]:


df1.TotalCharges= pd.to_numeric(df1.TotalCharges)


# In[ ]:


#Look at tenure to check the churn rate
tenure_churn_yes= df1[df1.Churn=='Yes'].tenure
tenure_churn_no= df1[df1.Churn=='No'].tenure

plt.xlabel("Number of months")
plt.ylabel("Number of customer")
plt.title("Customer Churn by Tenure")
plt.hist([tenure_churn_yes,tenure_churn_no], label=['Churn=Yes','Churn=No'])
plt.legend()

#We can see that loyal customer tend to leave less


# In[ ]:


mc_churn_yes= df1[df1.Churn=='Yes'].MonthlyCharges
mc_churn_no= df1[df1.Churn=='No'].MonthlyCharges

plt.xlabel("Number of months")
plt.ylabel("Number of customer")
plt.title("Customer Churn by Monthly Charges")
plt.hist([mc_churn_yes,mc_churn_no], label=['Churn=Yes','Churn=No'])
plt.legend()


# In[ ]:


for col in df1:
    if df1[col].dtypes=='object':
        print(f'{col}:{df1[col].unique()}')


# In[ ]:


#we will replace No internet service with No
df1.replace('No internet service', 'No', inplace=True)
df1.replace('No phone service', 'No', inplace=True)


# In[ ]:


yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns:
    df1[col].replace({'Yes': 1,'No': 0},inplace=True)


# In[ ]:


df1['gender'].replace({'Female':1,'Male':0},inplace=True)


# In[ ]:


df2 = pd.get_dummies(data=df1, columns=['InternetService','Contract','PaymentMethod'])
df2.columns


# In[ ]:


cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])


# In[ ]:


X = df2.drop('Churn',axis='columns')
y = df2['Churn']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)


# In[ ]:


X_train.shape


# In[ ]:


import tensorflow as tf
from tensorflow import keras


model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)


# In[ ]:


model.evaluate(X_test, y_test)


# In[ ]:


from sklearn.metrics import confusion_matrix , classification_report
yp = model.predict(X_test)
y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
print(classification_report(y_test,y_pred))


# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg= LogisticRegression()
logreg.fit(X_train,y_train)
y_pred_lr=logreg.predict(X_test)
print(classification_report(y_test,y_pred_lr))


# In[ ]:


from xgboost import XGBClassifier
XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
              eval_metric='logloss', gamma=0.1, gpu_id=-1, importance_type=None,
              interaction_constraints='', learning_rate=0.1, max_delta_step=0,
              max_depth=6, min_child_weight=10,
              monotone_constraints='()', n_estimators=100, n_jobs=4,
              num_parallel_tree=1, predictor='auto', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', use_label_encoder=False,
              validate_parameters=1, verbosity=None)
XGB.fit(X_train,y_train)
y_pred_xg=XGB.predict(X_test)
print(classification_report(y_test,y_pred_xg))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rbn=RandomForestClassifier(criterion='gini', max_depth=9, max_features='log2', n_estimators=300, min_samples_split=5)
rbn.fit(X_train,y_train)
y_pred_gd=rbn.predict(X_test)
print(classification_report(y_test,y_pred_gd))


# So what we have seen maybe Logistic regression is working better than normal we can see f1 score for is around 61% thats better than others so next what we wanna do is apply different methods to taclke the **imbalance** in data and again we will run these models
# 

# In[ ]:


#Now we will work with imablanced dataset techniques
#First lets check churn rate 
y_train.value_counts()


# ## Method 1 : Undersampling

# In[ ]:


#Lets define the models so we dont have to do copy paste again and again
def ANN(X_train, y_train, X_test, y_test, loss):
    model = keras.Sequential([
        keras.layers.Dense(26, input_dim=26, activation='relu'),
        keras.layers.Dense(15, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=100)
    
    
    print(model.evaluate(X_test, y_test))
    
    y_preds = model.predict(X_test)
    y_preds = np.round(y_preds)
    
    print("Classification Report: \n", classification_report(y_test, y_preds))
    
    return y_preds

def LogReg(X_train, X_test,y_train,y_test):
    logreg= LogisticRegression()
    logreg.fit(X_train,y_train)
    y_pred_lr=logreg.predict(X_test)
    print(classification_report(y_test,y_pred_lr))
    
    return y_pred_lr

def XGBClass(X_train, X_test,y_train,y_test):
    XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
              eval_metric='logloss', gamma=0.1, gpu_id=-1, importance_type=None,
              interaction_constraints='', learning_rate=0.1, max_delta_step=0,
              max_depth=6, min_child_weight=10,
              monotone_constraints='()', n_estimators=100, n_jobs=4,
              num_parallel_tree=1, predictor='auto', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', use_label_encoder=False,
              validate_parameters=1, verbosity=None)
    XGB.fit(X_train,y_train)
    y_pred_xg=XGB.predict(X_test)
    print(classification_report(y_test,y_pred_xg))
    
    return y_pred_xg

def Random(X_train, X_test,y_train,y_test):
    rbn=RandomForestClassifier(criterion='gini', max_depth=9, max_features='log2', n_estimators=300, min_samples_split=5)
    rbn.fit(X_train,y_train)
    y_pred_gd=rbn.predict(X_test)
    print(classification_report(y_test,y_pred_gd))
    
    return y_pred_gd


# In[ ]:


df_class_0= df2[df2.Churn==0]
df_class_1= df2[df2.Churn==1]
df_class_0.shape


# In[ ]:


df_class_1.shape


# In[ ]:


count_0, count_1= df2.Churn.value_counts()


# In[ ]:


#We are going to try undersampling for that we select only 1869 sample from df_class_0 using sample function of pandas

df_class_0=df_class_0.sample(count_1)
df_class_0.shape


# In[ ]:


#now we will concat both
df_under= pd.concat([df_class_0, df_class_1], axis=0)
df_under.shape


# In[ ]:


X = df_under.drop('Churn',axis='columns')
y = df_under['Churn']
zz
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)


# In[ ]:


y_preds = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy')


# So you can compare it with previous classification report we got better in f1 score of 1. lets check other models quickly

# In[ ]:


y_pred_lr= LogReg(X_train, X_test,y_train,y_test)


# In[ ]:


y_pred_xg= XGBClass(X_train, X_test,y_train,y_test)


# In[ ]:


y_pred_rd= Random(X_train, X_test,y_train,y_test)


# so with undersampling randomforest classifier gave better results. Now we will try to oversample using same sample method and see how it goes.
# ## Method 2: Oversampling

# In[ ]:


df_class_1_n=df_class_1.sample(count_0, replace=True)
df_over_sample= pd.concat([df_class_0, df_class_1_n], axis=0)
df_over_sample.shape


# In[ ]:


X = df_over_sample.drop('Churn',axis='columns')
y = df_over_sample['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)


# In[ ]:


y_pred_over= ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy')


# In[ ]:


print("Logistic Regression with oversample",  LogReg(X_train, X_test,y_train,y_test))

print("XGB with oversample", XGBClass(X_train, X_test,y_train,y_test) )

print("Random Forest classifier with oversample",Random(X_train, X_test,y_train,y_test))


# again Random Forest Classifier gave us much better result than others f1 score of both classes increased.

# Now we will use SMOTE to tackle imbalance and check its effect

# In[ ]:


#first we will do train and test split
X = df2.drop('Churn',axis='columns')
y = df2['Churn']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)


# In[ ]:


#We will apply smote only on training data
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(X_train, y_train)

y_sm.value_counts()


# In[ ]:


y_preds= ANN(X_sm, y_sm, X_test, y_test, 'binary_crossentropy')


# Not a good score compared to previous. It is an improvement from basic model but still not good

# In[ ]:


print("Logistic Regression with oversample",  LogReg(X_sm, X_test,y_sm,y_test))

print("XGB with oversample", XGBClass(X_sm, X_test,y_sm,y_test) )

print("Random Forest classifier with oversample",Random(X_sm, X_test,y_sm,y_test))


# We used SMOTE only on training function. One of the reason we have seen high f1 score above is beacuse we used oversampling on data itself and then did split so overlapping might be possible. We will try two more methods before we end

# In[ ]:


from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
#We will try SMOTEENN. This method combines the SMOTE ability to generate synthetic examples for minority class 
#and ENN ability to delete some observations from both classes that are identified as having different class between the observation’s class and its K-nearest neighbor majority class.
smoten = SMOTEENN()
x_smoten, y_smoten = smoten.fit_resample(X_train, y_train)


# In[ ]:


y_preds= ANN(x_smoten, y_smoten, X_test, y_test, 'binary_crossentropy')


# In[ ]:


print("Logistic Regression with oversample",  LogReg(x_smoten, X_test,y_smoten,y_test))

print("XGB with oversample", XGBClass(x_smoten, X_test,y_smoten,y_test) )

print("Random Forest classifier with oversample",Random(x_smoten, X_test,y_smoten,y_test))


# In[ ]:


#Oversample using Adaptive Synthetic (ADASYN) algorithm.
#This method is similar to SMOTE but it generates different number of samples depending on an estimate of the local distribution of the class to be oversampled.
ada = ADASYN(random_state=101)
x_res, y_res = ada.fit_resample(X_train, y_train)


# In[ ]:


print("Logistic Regression with oversample",  LogReg(x_res, X_test,y_res,y_test))

print("XGB with oversample", XGBClass(x_res, X_test,y_res,y_test) )

print("Random Forest classifier with oversample",Random(x_res, X_test,y_res,y_test))


# In[ ]:


y_preds= ANN(x_res, y_res, X_test, y_test, 'binary_crossentropy')


# Now when I go back I think that high f1 score might be cuz of overlapping between training and testing data while sampling but It seems random forest classifier is performing better than others.

# In[ ]:




