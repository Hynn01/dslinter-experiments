#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection
# 
# ---
# 
# **Table of Contents** 
# 
# [1.0 Objectives](#1.0-Objectives)  
# [2.0 Import Library](#2.0-Import-Library)  
# [3.0 Set Constant and Default Settings](#3.0-Set-Constant-and-Default-Settings)  
# [4.0 Load Dataset](#4.0-Load-Dataset)  
# [5.0 Split dataset into train and test set](#5.0-Split-dataset-into-train-and-test-set)  
# [6.0 Exploratory Data Analysis (EDA)](#6.0-Exploratory-Data-Analysis-%28EDA%29)  
# &nbsp; &nbsp; &nbsp; [6.1 Statistics Summary of Dataset](#6.1-Statistics-Summary-of-Dataset)  
# &nbsp; &nbsp; &nbsp; [6.2 Detect for Outliers](#6.2-Detect-for-Outliers)   
# &nbsp; &nbsp; &nbsp; [6.3 Data Cleaning](#6.3-Data-Cleaning)   
# &nbsp; &nbsp; &nbsp; [6.4 Feature Scaling](#6.4-Feature-Scaling)   
# &nbsp; &nbsp; &nbsp; [6.5 Full Transformation](#6.5-Full-Transformation)   
# [7.0 Model Evaluation Metrics](#7.0-Model-Evaluation-Metrics)   
# [8.0 Machine Learning Model](#8.0-Machine-Learning-Model)  
# &nbsp; &nbsp; &nbsp; [8.1 Benchmark Model - Logistic Regression](#8.1-Benchmark-Model-\--Logistic-Regression)  
# &nbsp; &nbsp; &nbsp; [8.2 Other Models](#8.2-Other-Models)  
# &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [8.2.1 Stochastic Gradient Descent (SGD) Classifier](#8.2.1-Stochastic-Gradient-Descent-%28SGD%29-Classifier)  
# &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [8.2.2 Random Forest Classifier](#8.2.2-Random-Forest-Classifier)  
# &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [8.2.3 Support Vector Machine Classifier](#8.2.3-Support-Vector-Machine-Classifier)  
# &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [8.2.4 Naive Bayes](#8.2.4-Naive-Bayes)  
# &nbsp; &nbsp; &nbsp; [8.3 Model Selection](#8.3-Model-Selection)  
# &nbsp; &nbsp; &nbsp; [8.4 Fine-Tuning Model](#8.4-Fine-Tuning-Model)  
# &nbsp; &nbsp; &nbsp; [8.5 Resample on Dataset](#8.5-Resample-on-Dataset)  
# &nbsp; &nbsp; &nbsp; [8.6 Initialise Class Weight on Model](#8.6-Initialise-Class-Weight-on-Model)  
# [9.0 Conclusion](#9.0-Conclusion)  
# 
# ---

# # 1.0 Objectives
# 
# - To build a model that detect credit card fraud

# # 2.0 Import Library

# In[ ]:


# System
import os

# Parallel
from joblib import effective_n_jobs

# EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Preprocessing data
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Metrics
from sklearn.metrics import f1_score, recall_score

# Model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Fine-tune
from sklearn.model_selection import RandomizedSearchCV

# Resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# # 3.0 Set Constant and Default Settings

# In[ ]:


plt.rcParams['figure.dpi'] = 150
sns.set_style('dark')


# In[ ]:


base_dir = os.path.join('/', 'kaggle', 'input', 'creditcardfraud')

# Check if kaggle env or local env
is_kaggle = os.path.exists(base_dir)

dataset_path = os.path.join(base_dir if is_kaggle else '', 'creditcard.csv')


# In[ ]:


n_thread =  effective_n_jobs(-1)

if not is_kaggle:
    n_thread =  int(effective_n_jobs(-1) * 0.7)


# # 4.0 Load Dataset

# In[ ]:


df = pd.read_csv(dataset_path)


# # 5.0 Split dataset into train and test set

# In[ ]:


X_df = df.drop(['Class'], axis=1)
y_df = df['Class']


# In[ ]:


train_test_sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in train_test_sss.split(X_df, y_df):
    train_df = df.loc[train_index]
    test_df = df.loc[test_index]


# In[ ]:


X_train_df = train_df.drop(['Class'], axis=1)
y_train_df = train_df['Class']


# In[ ]:


train_val_sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, val_index in train_val_sss.split(X_train_df, y_train_df):
    train_df = df.loc[train_index]
    val_df = df.loc[val_index]


# # 6.0 Exploratory Data Analysis (EDA)

# ## 6.1 Statistics Summary of Dataset

# In[ ]:


train_df.head()


# In[ ]:


train_df.columns


# Due to privacy issues, the dataset feature name has been masked and been named 'V1', 'V2' and etc.

# In[ ]:


train_df.info()


# In[ ]:


df.describe().T


# As stated from the dataset author, the feature in the dataset is obtained from PCA. The only features that not obtained from PCA are 'Amount' and 'Time'. Standard scaler is required to apply on these two features. It is to avoid the model to have a false sense of feature 'Amount' and 'Time' is more significant than the rest of features when model being trained.

# In[ ]:


label_weight_perc = train_df['Class'].value_counts(normalize=True) * 100

label_weight_perc


# Another interesting fact is that the dataset label is extremely imbalanced. 99.83% of the instances in the dataset is not fraud while only 0.17% of the instances are fraud. This will lead the model training extremely difficult and appropriate metrics to measure the model is required.

# In[ ]:


train_df.corr()['Class'].sort_values()


# ## 6.2 Detect for Outliers

# In[ ]:


sns.boxplot(data=train_df, y='Amount')

plt.show()


# From above graph, we can see the box plot for the feature 'Amount' contains outlier

# In[ ]:


feature_outlier = dict()

class Outlier:
    def __init__(self, q1, q3):
        self.q1 = q1
        self.q3 = q3
        self.iqr = q3 - q1
    
    def get_outlier_boundary(self):
        lower_fence = self.q1 - 1.5 * self.iqr
        upper_fence = self.q3 + 1.5 * self.iqr
        
        return lower_fence, upper_fence


def filter_outlier(df, cols=[]):
    if 'is_outlier' not in df.columns:
        df['is_outlier'] = (False) * len(df)
        
    for col in cols:
        if col in feature_outlier.keys():
            outlier = feature_outlier[col]
        else:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            
            outlier = Outlier(q1, q3)
            feature_outlier[col] = outlier
             
        lower_fence, upper_fence = outlier.get_outlier_boundary()
        
        outlier = (df[col] < lower_fence) | (df[col] > upper_fence)
        
        df['is_outlier'] = outlier | df['is_outlier']
        
    df = df[~df['is_outlier']]
    df = df.drop(['is_outlier'], axis=1)
             
    return df


# In[ ]:


train_df = filter_outlier(train_df, cols=['Amount'])


# ## 6.3 Data Cleaning

# In[ ]:


train_df.isnull().sum()


# Data cleaning is not required as the dataset has been cleaned by the dataset author.

# ## 6.4 Feature Scaling

# Applying standardisation to the feature 'Amount' and 'Time'

# In[ ]:


std_feat = ['Amount', 'Time']
std_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])


# ## 6.5 Full Transformation

# In[ ]:


full_pipeline = ColumnTransformer([
    ('std_feat', std_pipeline, std_feat)
], remainder='passthrough')


# In[ ]:


X_train = train_df.drop(['Class'], axis=1)
y_train = train_df['Class']

X_train = full_pipeline.fit_transform(X_train)


# In[ ]:


X_val = val_df.drop(['Class'], axis=1)
y_val = val_df['Class']

X_val = full_pipeline.transform(X_val)


# In[ ]:


X_test = test_df.drop(['Class'], axis=1)
y_test = test_df['Class']

X_test = full_pipeline.transform(X_test)


# # 7.0 Model Evaluation Metrics

# Since the dataset label is extremely imbalanced, only 0.18% of instances in the dataset is fraud. Selecting an appropriate evaluation metrics is crucial as standard metrics work well on balanced dataset. 
# 
# For example, using the accuracy as metrics for this dataset. The model can just predict all the instances in the dataset as non-fraud and accuracy of the model still be 99.82%. As 99.82% of the instances in the dataset is non-fraud.
# 
# Hence, we need to select 'recall' and 'f1-score' as the metrics to evaluate the model.

# $$Recall = \frac{True~Positive}{True~Positive + False~Negative}$$

# $$F1~Score = \frac{2 \times Precision \times Recall}{Precision + Recall}$$

# In[ ]:


model_eval = {
    'model': [],
    'recall': [],
    'f1_score': []
}

def add_model_eval(model, recall, f1_score):
    model_eval['model'].append(model)
    model_eval['recall'].append(f'{recall: .2f}')
    model_eval['f1_score'].append(f'{f1_score: .2f}')
    
def view_models_eval(sort=False):
    eval_df = pd.DataFrame(model_eval)
    
    if sort:
        eval_df = eval_df.sort_values(by=['recall', 'f1_score'], ascending=[False, False])
    
    display(eval_df.style.hide_index())


# # 8.0 Machine Learning Model

# ## 8.1 Benchmark Model - Logistic Regression

# The benchmark model for classify credit card instance is fraud or non-fraud will be logistic regression. A logistic regression is the most common model for binary classification.

# In[ ]:


log_reg = LogisticRegression(random_state=42, verbose=1)
log_reg.fit(X_train, y_train)


# In[ ]:


y_pred = log_reg.predict(X_val)

add_model_eval('logistic regression', recall_score(y_val, y_pred), f1_score(y_val, y_pred))


# In[ ]:


view_models_eval()


# Based on logistic regression, the recall and f1_score are 0.51 and 0.63 respectively. With a recall of 0.51, the model able to detect about half of the fraud instances correctly. There still room to improve the metric score. Let try other model with default parameter first and compare to logistic regression before fine-tune the logistic regression.

# ## 8.2 Other Models

# Below the list of model will be trained to classify the credit card dataset as fraud or non-fraud:
# - Stochastic Gradient Descent (SGD) Classifier
# - Random Forest Classifier
# - Support Vector Machine Classifier
# - Naive Bayes

# ### 8.2.1 Stochastic Gradient Descent (SGD) Classifier

# In[ ]:


sgd_clf = SGDClassifier(random_state=42, verbose=1)
sgd_clf.fit(X_train, y_train)


# In[ ]:


y_pred = sgd_clf.predict(X_val)

add_model_eval('sgd classifier', recall_score(y_val, y_pred), f1_score(y_val, y_pred))


# In[ ]:


view_models_eval()


# ### 8.2.2 Random Forest Classifier

# In[ ]:


forest_clf = RandomForestClassifier(random_state=42, verbose=2, n_jobs=n_thread)
forest_clf.fit(X_train, y_train)


# In[ ]:


y_pred = forest_clf.predict(X_val)

add_model_eval('random forest classifier', recall_score(y_val, y_pred), f1_score(y_val, y_pred))


# In[ ]:


view_models_eval()


# ### 8.2.3 Support Vector Machine Classifier

# In[ ]:


svm_clf = SVC(random_state=42, verbose=2)
svm_clf.fit(X_train, y_train)


# In[ ]:


y_pred = svm_clf.predict(X_val)

add_model_eval('support vector machine classifier', recall_score(y_val, y_pred), f1_score(y_val, y_pred))


# In[ ]:


view_models_eval()


# ### 8.2.4 Naive Bayes

# In[ ]:


nb = GaussianNB()
nb.fit(X_train, y_train)


# In[ ]:


y_pred = svm_clf.predict(X_val)

add_model_eval('naive bayes', recall_score(y_val, y_pred), f1_score(y_val, y_pred))


# In[ ]:


view_models_eval()


# ## 8.3 Model Selection

# In[ ]:


view_models_eval(sort=True)


# From the above table, we can know that random forest classifier model has the highest recall and f1_score by using default parameter. Thus, we will use random forest classifier to predict fraud and non-fraud for the dataset.

# ## 8.4 Fine-Tuning Model

# Now, let try to fine-tune the parameter of the random forest classifier.

# In[ ]:


# clf = classifier
# ft = fine-tune
forest_clf_ft = RandomForestClassifier(random_state=42, n_jobs=n_thread)

param_grid = {
    'bootstrap': [True, False],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [100, 200, 300, 400, 500]
}

random_search = RandomizedSearchCV(forest_clf_ft, 
                                   param_grid, 
                                   random_state=42,
                                   cv=2,
                                   n_iter=10, 
                                   scoring='recall',
                                   verbose=2)

random_search.fit(X_train, y_train)


# In[ ]:


forest_clf_best_params = random_search.best_params_

forest_clf_best_params


# In[ ]:


y_pred = random_search.predict(X_val)

add_model_eval('random forest classifier with fine-tune', recall_score(y_val, y_pred), f1_score(y_val, y_pred))


# In[ ]:


view_models_eval()


# After fine-tune the random forest classifier, the model has a slightly increase on recall than default parameter.

# ## 8.5 Resample on Dataset

# Besides, getting the optimum parameter for the random forest classifier model. We also can leverage the resampling technique on the dataset to make the label balance.
# 
# First, use the SMOTE to oversample of the minority class Next, use random undersampling to reduce the number of instances of majority class. In the end, the ratio of non-fraud and fraud will be $1 : 1$

# In[ ]:


pd.Series(y_train).value_counts(normalize=True)


# Before resample, the ratio of non-fraud to fraud is $99 : 1$

# In[ ]:


oversampling = SMOTE(random_state=42)
undersampling = RandomUnderSampler(random_state=42)
steps = [('o', oversampling), ('u', undersampling)]
pipeline = Pipeline(steps=steps)


# In[ ]:


X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train, y_train)


# In[ ]:


pd.Series(y_train_resampled).value_counts(normalize=True)


# After resample, the ratio of non-fraud to fraud is $1 : 1$

# In[ ]:


# clf = classifier
# ft = fine-tune
# rs = resample
forest_clf_ft_rs = RandomForestClassifier(**forest_clf_best_params,
                                       random_state=42, 
                                       verbose=2, 
                                       n_jobs=n_thread)

forest_clf_ft_rs.fit(X_train_resampled, y_train_resampled)


# In[ ]:


y_pred = forest_clf_ft_rs.predict(X_val)

add_model_eval('random forest classifier with fine-tune and resample', recall_score(y_val, y_pred), f1_score(y_val, y_pred))


# In[ ]:


view_models_eval()


# With the resampling technique apply to the train dataset, the model has recall score has improved from 0.80 to 0.82 while f1 decrease slightly to 0.85. Despite that, it is worth the trade.

# ## 8.6 Initialise Class Weight on Model

# Lastly, let try to specify the class weight on the model to see whether any improvement will be made or not on the original train dataset (i.e. not the dataset been resampled)

# In[ ]:


# clf = classifier
# ft = fine-tune
# cw = class weight
forest_clf_ft_cw = RandomForestClassifier(**forest_clf_best_params,
                                       class_weight='balanced',
                                       random_state=42, 
                                       verbose=2, 
                                       n_jobs=n_thread)

forest_clf_ft_cw.fit(X_train, y_train)


# In[ ]:


y_pred = forest_clf_ft_cw.predict(X_val)

add_model_eval('random forest classifier with fine-tune and class weight', recall_score(y_val, y_pred), f1_score(y_val, y_pred))


# In[ ]:


view_models_eval()


# Despite having set the class weight on the random forest classifier model, the model recall and f1 score been decreased

# # 9.0 Conclusion

# In[ ]:


view_models_eval(sort=True)


# After trying searching and tweaking different models, the model that score the highest for both recall and f1 score on the validation set will be random forest classifier with fine-tune and resample.

# In[ ]:


y_pred = forest_clf_ft_rs.predict(X_test)

recall_score(y_test, y_pred), f1_score(y_test, y_pred)


# The model score 0.94 and 0.94 for recall and f1 score respectively on the test dataset. 
