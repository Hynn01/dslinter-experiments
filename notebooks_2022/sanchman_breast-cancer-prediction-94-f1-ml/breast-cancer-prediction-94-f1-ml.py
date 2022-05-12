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


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/breast-cancer-prediction-dataset/Breast_cancer_data.csv')


# In[ ]:


df.head()


# # EDA 

# In[ ]:


df.describe()


# In[ ]:


df.duplicated().sum()


# In[ ]:


df.isnull().sum()


# In[ ]:


cols = list(df.columns)
print(cols)


# In[ ]:


sns.countplot(x='diagnosis', data=df, palette='GnBu')
plt.title('Diagnosis Displot')
plt.show()


# Our main metric to measure the performance will be f1 along with precision and recall as there is a little bit of data imbalance

# In[ ]:


corr = df.corr(method = 'spearman')
plt.figure(figsize=(20,6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues')
plt.title('Spearman Correlation Heatmap')
plt.show()


# No two columns are highly correlated so no need to drop any column.

# In[ ]:


figure = plt.figure(figsize=(20,10))
sns.pairplot(df, hue='diagnosis', palette='GnBu')
plt.show()


# Data looks separable with high accuracy

# In[ ]:


fig, axes = plt.subplots(5, 3, figsize=(20,25))
for i, col in zip(range(5), cols):
    sns.stripplot(ax=axes[i][0], x='diagnosis', y=col, data=df, palette='GnBu', jitter=True)
    axes[i][0].set_title(f'{col} Stripplot')
    sns.histplot(ax=axes[i][1], x=col, data=df, kde=True, bins=10, palette='GnBu', hue='diagnosis', multiple='dodge')
    axes[i][1].set_title(f'{col} Displot')
    sns.boxplot(ax=axes[i][2], x='diagnosis', y=col, data=df, palette='GnBu', hue='diagnosis')
    axes[i][2].set_title(f'{col} Boxplot')


# Let's replace the outliers with the upper and lower bounds of the interquartile range

# In[ ]:


def outlier_limits(df, col_name, q1 = 0.25, q3 = 0.75):
    quartile1 = df[col_name].quantile(q1)
    quartile3 = df[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_limits(df, variable, q1 = 0.25, q3 = 0.75):
    low_limit, up_limit = outlier_limits(df, variable, q1 = q1, q3 = q3)
    df.loc[(df[variable] < low_limit), variable] = low_limit
    df.loc[(df[variable] > up_limit), variable] = up_limit
    
for variable in cols:
    replace_with_limits(df, variable)


# In[ ]:


fig, axes = plt.subplots(1, 5, figsize=(25,6))
for i, col in zip(range(5), cols):
    sns.boxplot(ax=axes[i], x='diagnosis', y=col, data=df, palette='GnBu', hue='diagnosis')
    axes[i].set_title(f'{col} Boxplot')


# # Preprocessing

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=cols)
df_scaled.head()


# In[ ]:


X = df_scaled.iloc[:,:5]
y = df_scaled['diagnosis']
X.head()


# # Machine Learning

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


# In[ ]:


scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    }


# In[ ]:


def fit(clf, params, cv=10, X_train=X_train, y_train=y_train):
    grid = GridSearchCV(clf, params, cv=KFold(n_splits=cv), n_jobs=1, verbose=1, return_train_score=True, scoring=scoring, refit='f1') #verbose and n_jobs help us see the computation time and score of a cv. Higher the value of verbose, more the information printed out.
    grid.fit(X_train, y_train)
    return grid

def make_predictions(model, X_test=X_test):
    return model.predict(X_test)

def best_scores(model):
    best_mean_f1 = max(list(model.cv_results_['mean_test_f1']))
    mean_f1_index = list(model.cv_results_['mean_test_f1']).index(best_mean_f1)
    print(f'The best parameters are: {model.best_params_}')
    print('Mean Test Cross Validation Scores for different metrics: (corresponding to best mean f1)')
    print('The best score that we get is (Accuracy): ' + str(model.cv_results_['mean_test_accuracy'][mean_f1_index]))
    print('The best score that we get is (Precision): ' + str(model.cv_results_['mean_test_precision'][mean_f1_index]))
    print('The best score that we get is (Recall): ' + str(model.cv_results_['mean_test_recall'][mean_f1_index]))
    print(f'The best score that we get is (F1 Score): {best_mean_f1}')
    return None

def plot_confusion_matrix(y_pred):
    print('00: True Negatives\n01: False Positives\n10: False Negatives\n11: True Positives\n')
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap='GnBu', alpha=0.75)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='large') 
    plt.xlabel('Predictions', fontsize=14)
    plt.ylabel('Actuals', fontsize=14)
    plt.title('Confusion Matrix', fontsize=14)
    plt.show()
    return None

def check_scores(y_pred):
    print('Precision: %.3f' % precision_score(y_test, y_pred))
    print('Recall: %.3f' % recall_score(y_test, y_pred))
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
    print('F1 Score: %.3f' % f1_score(y_test, y_pred))
    return None


# In[ ]:


import warnings
warnings.filterwarnings('always')


# ## Logistic Regression

# In[ ]:


lr_params = {'C':[0.001,.009,0.01,.09,1,5,10,25], 'penalty':['l1', 'l2']} #lasso and ridge regression
lr_clf = LogisticRegression(solver='saga', max_iter=5000)
lr_model = fit(lr_clf, lr_params)


# In[ ]:


best_scores(lr_model)


# In[ ]:


lr_y_pred = make_predictions(lr_model)
check_scores(lr_y_pred)


# In[ ]:


plot_confusion_matrix(lr_y_pred)


# In[ ]:


lr_feature_scores = lr_model.best_estimator_.coef_[0].tolist()
lr_fi = pd.DataFrame({'Feature': X.columns, 'Feature Importance': lr_feature_scores})
plt.figure(figsize=(10,6))
sns.barplot(x='Feature Importance', y='Feature', data=lr_fi, palette='GnBu', )
plt.show()


# ## LDA

# In[ ]:


lda_params = {'solver': ['svd', 'eigen']}
lda_clf = LDA()
lda_model = fit(lda_clf, lda_params)


# In[ ]:


best_scores(lda_model)


# In[ ]:


lda_y_pred = make_predictions(lda_model)
check_scores(lda_y_pred)


# In[ ]:


plot_confusion_matrix(lda_y_pred)


# ## SVM

# In[ ]:


svm_params = {'C':[1,10,100,1000], 'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}
svm_clf = SVC()
svm_model = fit(svm_clf, svm_params)


# In[ ]:


best_scores(svm_model)


# In[ ]:


svm_y_pred = make_predictions(svm_model)
check_scores(svm_y_pred)


# In[ ]:


plot_confusion_matrix(svm_y_pred)


# ## K-NNs

# In[ ]:


knns_params = {'n_neighbors': list(range(1, 31)), 'weights': ['uniform', 'distance'], 
               'metric': ['euclidean', 'manhattan']}
knns_clf = KNeighborsClassifier()
knns_model = fit(knns_clf, knns_params)


# In[ ]:


best_scores(knns_model)


# In[ ]:


knns_y_pred = make_predictions(knns_model)
check_scores(knns_y_pred)


# In[ ]:


plot_confusion_matrix(knns_y_pred)

