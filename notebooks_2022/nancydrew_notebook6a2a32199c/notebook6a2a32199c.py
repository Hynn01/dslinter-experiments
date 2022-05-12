#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report, precision_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Importing the dataset

# In[ ]:


df = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
df.head()


# In[ ]:


df.info()


# In[ ]:


df = df.drop('Unnamed: 32', axis=1)
df = df.drop('id', axis=1)


# In[ ]:


sns.countplot(df['diagnosis'])


# The output variables donot have a common distribution hence while splitting them I will use stratify

# # Cluster analysis with the data

# In[ ]:


train_data = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")


# In[ ]:


train_data = train_data.drop('id',axis=1)
train_data = train_data.drop('Unnamed: 32',axis=1)

train_data['diagnosis'] = train_data['diagnosis'].map({'M':1,'B':0})

# Scaling the dataset
datas = pd.DataFrame(preprocessing.scale(train_data.iloc[:,1:32]))
datas.columns = list(train_data.iloc[:,1:32].columns)

X = datas
y = train_data['diagnosis']


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
Y = pca.fit_transform(X)


# In[ ]:


### KMeans cluster analysis
datas['diagnosis'] = train_data['diagnosis']


from sklearn.cluster import KMeans

kmns = KMeans(n_clusters=2, algorithm='full')
KY = kmns.fit_predict(X)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.scatter(Y[:,0],Y[:,1],  c=KY, cmap = "jet", edgecolor = "None", alpha=0.35)
ax1.set_title('k-means clustering plot')

ax2.scatter(Y[:,0],Y[:,1],  c = datas['diagnosis'], cmap = "jet", edgecolor = "None", alpha=0.35)
ax2.set_title('Actual clusters')


# In[ ]:


### Agglomerative Clustering analysis

from sklearn.cluster import AgglomerativeClustering
aggC = AgglomerativeClustering()
kY = aggC.fit_predict(X)


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)


ax1.scatter(Y[:,0],Y[:,1],  c=kY, cmap = "jet", edgecolor = "None", alpha=0.35)
ax1.set_title('Hierarchical clustering plot')

ax2.scatter(Y[:,0],Y[:,1],  c = datas['diagnosis'], cmap = "jet", edgecolor = "None", alpha=0.35)
ax2.set_title('Actual clusters')


# In[ ]:


cols = ['diagnosis',
        'radius_mean', 
        'texture_mean', 
        'perimeter_mean', 
        'area_mean', 
        'smoothness_mean', 
        'compactness_mean', 
        'concavity_mean',
        'concave points_mean', 
        'symmetry_mean', 
        'fractal_dimension_mean']

sns.pairplot(data=df[cols], hue='diagnosis', palette='RdBu')


# 1. Perimeter mean, Area Mean and Radius Mean are implying multicollinearity
# 2. Concavity_mean, concavePoints_means, Compactness_mean imply multicollinearity

# In[ ]:


corr_mat = df.corr()
cols = corr_mat.index

f, ax = plt.subplots(figsize=(25, 25))
hm = sns.heatmap(corr_mat,vmin=-1, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, yticklabels=cols.values, xticklabels=cols.values)


# 1. There is high correlation between radius_mean with perimeter_mean and area_mean.(similar for radius_se,perimeter_se, and area_se and radius_worst,perimeter_worst,area_worst)
# 2. There is high correlation between compactness, concavity and concave_points
# 3. There is high correlation between mean values and worst values

# In[ ]:


### Standardizing the data

X = df[cols].values

y = df['diagnosis'].map({'M':1,'B':0})


# Using stratify to split the data into training and test set

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state = 21,stratify=y)


# ## Running a bunch of classification models with default values to pick out the classifier. We will fine tune the model with the most accuracy

# In[ ]:


key = ['LogisticRegression','KNeighborsClassifier','SVC','DecisionTreeClassifier','RandomForestClassifier','GradientBoostingClassifier','AdaBoostClassifier','XGBClassifier']
value = [LogisticRegression(),KNeighborsClassifier(n_neighbors = 2),SVC(random_state=15),DecisionTreeClassifier(random_state=10), RandomForestClassifier(n_estimators=60, random_state=0), GradientBoostingClassifier(random_state=20), AdaBoostClassifier(), xgb.XGBClassifier(random_state=0,booster="gbtree")]

models = dict(zip(key,value))
models


# In[ ]:


scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

accurarcy = []

for name,algo in models.items():
    model = algo
    model.fit(X_train_scaled,y_train)
    predict = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, predict)
    accurarcy.append(acc)

performance = dict(zip(key,accurarcy))


# In[ ]:


performance


# SVC and Logistic Regression seem to have high performance(other models have similar or lower performance).

# In[ ]:


### KFOLD cross validation and hyperparameter tunning for LOGISTIC REGRESSION
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

penalty_values = ['l1', 'l2', 'elasticnet']
class_weight_values = ['balanced']
solver_values = ['liblinear']

param_grid = dict(penalty=penalty_values,class_weight=class_weight_values,solver=solver_values)
model = LogisticRegression()

kFold = KFold(n_splits=10)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy',cv=kFold) 

## Fit the model 
grid_result = grid.fit(X_train_scaled,y_train)
print("Best score: %f" % (grid_result.best_score_))
print(grid_result.best_params_)


# In[ ]:


scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(class_weight='balanced', penalty='l2', solver='liblinear')
model.fit(X_train_scaled,y_train)

predictions_logisticRegression = model.predict(X_test_scaled)
print("Accuracy score %f" % accuracy_score(y_test, predictions_logisticRegression))


# In[ ]:


### KFOLD cross validation and hyperparameter tunning for SVC
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']

param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC()

kFold = KFold(n_splits=10)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kFold)

## Fit the model
grid_result = grid.fit(rescaledX,y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(C=1.5, kernel='rbf')
model.fit(X_train_scaled,y_train)

predictions_SVC = model.predict(X_test_scaled)
print("Accuracy score %f" % accuracy_score(y_test, predictions_SVC))


# In[ ]:


## Having a look at misclassified points for SVC

from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, predictions_SVC, digits=3))

cfm = confusion_matrix(y_test, predictions_SVC)

true_negative = cfm[0][0]
false_positive = cfm[0][1]
false_negative = cfm[1][0]
true_positive = cfm[1][1]

print('Confusion Matrix: \n', cfm, '\n')

print('True Negative:', true_negative)
print('False Positive:', false_positive)
print('False Negative:', false_negative)
print('True Positive:', true_positive)

