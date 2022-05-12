#!/usr/bin/env python
# coding: utf-8

# # Try lots of models, features

# **Mobile Price Classification**
# > classify mobile price range

# <a id="Import_libraries"></a>
# # Import libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import accuracy_score 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans


# In[ ]:


# Show plots in jupyter lab
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id="Load_data"></a>
# # Load data

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


df_org = pd.read_csv("/kaggle/input/mobile-price-classification/train.csv")
df = df_org.copy()


# <a id="3"></a>
# # Look at data and Data cleansing

# In[ ]:


print(df.shape)


# > Rows: 2000<br>
# Columns: 21

# In[ ]:


pd.set_option('display.max_columns', None)
df.head(10)


# # Make price range 0 and 1

# In[ ]:


def filter(x):
    if x > 1:
        x = 1
        return x
    else:
        return 0


# In[ ]:


df.price_range = df.price_range.apply(filter)


# In[ ]:


df.head(3)


# In[ ]:


df.columns


# > All columns are usefull

# In[ ]:


df.info()


# > Types are ok

# In[ ]:


df.duplicated().sum()


# In[ ]:


df.isnull().sum()


# > there is no Null item 

# In[ ]:


df.describe()


# In[ ]:


# Return unbiased kurtosis over requested axis using Fisher's definition of kurtosis (kurtosis of normal == 0.0).
# The result is normalized by N-1
kurt = df.kurt()[:]
kurt


# > It's ok 

# <a id='Data_Visualization'></a>
# # Data Visualization

# In[ ]:


# Visualizing the distribution for every "feature"
df.hist(edgecolor="black", linewidth=1.2, figsize=(20, 20))
plt.show()


# In[ ]:


# correlation matrix
plt.figure(figsize=(20, 10))
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[ ]:


plt.figure(figsize=(10,8))
ax = sns.countplot(x = df["price_range"])


# In[ ]:


for col in df.columns[:12]:
    plt.figure(figsize=(15,8))
    sns.violinplot(x="price_range", y=col, data=df)


# In[ ]:


for col in df.columns[12:-1]:
    plt.figure(figsize=(15,8))
    sns.violinplot(x="price_range", y=col, data=df)


# # Preprocessing

# In[ ]:


df_copy = df.copy()


# In[ ]:


y = df_copy.pop('price_range')
X = df_copy


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_train.shape)


# In[ ]:


y_train.head()


# In[ ]:


X_train.head(3)


# In[ ]:


scaler_1 = StandardScaler()


# In[ ]:


fitter_1 = scaler_1.fit(X_train)


# In[ ]:


X_train = fitter_1.transform(X_train)
X_test = fitter_1.transform(X_test)


# In[ ]:


X_train = pd.DataFrame(data=X_train, columns=X.columns)
X_test = pd.DataFrame(data=X_test, columns=X.columns)
# y_train = pd.DataFrame(data=y_train, columns=['price'])
# y_test = pd.DataFrame(data=y_test, columns=['price'])


# In[ ]:


X_train.head(3)


# In[ ]:


y_train.head(3)


# In[ ]:


X_test.head()


# <a id="Model_Function"></a>
# # Model function

# In[ ]:


def LR(X_train, y_train, X_test, y_test, i):
    LR = LogisticRegression()
    LR = LR.fit(X_train, y_train)
    yhat = LR.predict(X_test)
    acc = accuracy_score(y_test, yhat)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, yhat)
    acc_auc = metrics.auc(fpr, tpr)
    if isinstance(i, int):
        #Forward, Backward selection we need acc_auc
        if i == 22:
            return (acc_auc)
        #predict test data we need model
        elif i == 23:
            return LR
    #all features ,pca
    else:
        message = f"Accuracy of sklearn's Logistic Regression Classifier with {i}: {acc}, acc_auc: {acc_auc}"
        return (message, yhat)


# <a id="Fauture_selection"></a>
# # Fauture selection 

# <a id="With_all_feautres"></a>
# # With all feautres 

# In[ ]:


message, yhat = LR(X_train, y_train, X_test, y_test, "all_features")
print(message)
print(classification_report(y_test, yhat))


# In[ ]:


cm = confusion_matrix(y_test, yhat)

df1 = pd.DataFrame(columns=["0","1"], index= ["0","1"], data= cm )

f,ax = plt.subplots(figsize=(6,6))

sns.heatmap(df1, annot=True,cmap="Greens", fmt= '.0f',
            ax=ax,linewidths = 5, cbar = False,annot_kws={"size": 16})
plt.xlabel("Predicted Label")
plt.xticks(size = 12)
plt.yticks(size = 12, rotation = 0)
plt.ylabel("True Label")
plt.title("Confusion Matrix", size = 12)
plt.show()


# # Forward selection

# In[ ]:


def forward(X_train, y_train, X_test, y_test, best_cols, all_cols):
    init_acc_auc = 0.50
    for col in all_cols:
        best_cols.append(col)
        X_train_f = pd.DataFrame(data=X_train, columns=best_cols)
        X_test_f = pd.DataFrame(data=X_test, columns=best_cols)
        acc_auc = LR(X_train_f, y_train, X_test_f, y_test, i=22)
        if acc_auc > init_acc_auc:
            init_acc_auc = acc_auc
        else:
            best_cols.pop()
    return(f'result forward selection=> best columns: {best_cols} with auc: {init_acc_auc}', best_cols)


# In[ ]:


best_cols, all_cols = [], X.columns.to_list()
forward_col, best_cols_f = forward(X_train, y_train, X_test, y_test, best_cols, all_cols)
print(forward_col)


# In[ ]:


X_train_f = pd.DataFrame(data=X_train, columns=best_cols_f)
X_test_f = pd.DataFrame(data=X_test, columns=best_cols_f)
message, yhat = LR(X_train_f, y_train, X_test_f, y_test, "forward selection")
print(message)
print(classification_report(y_test, yhat))


# In[ ]:


cm = confusion_matrix(y_test, yhat)

df1 = pd.DataFrame(columns=["0","1"], index= ["0","1"], data= cm )

f,ax = plt.subplots(figsize=(6,6))

sns.heatmap(df1, annot=True,cmap="Greens", fmt= '.0f',
            ax=ax,linewidths = 5, cbar = False,annot_kws={"size": 16})
plt.xlabel("Predicted Label")
plt.xticks(size = 12)
plt.yticks(size = 12, rotation = 0)
plt.ylabel("True Label")
plt.title("Confusion Matrix", size = 12)
plt.show()


# # Backward selection

# In[ ]:


def backward(X_train, y_train, X_test, y_test, all_cols):
    init_acc_auc = LR(X_train, y_train, X_test, y_test, i=22)
    for i in range(len(all_cols)-1):
        col = all_cols.pop(0)
        X_train_b = pd.DataFrame(data=X_train, columns=all_cols)
        X_test_b = pd.DataFrame(data=X_test, columns=all_cols)
        acc_auc = LR(X_train_b, y_train, X_test_b, y_test, i=22)
        if acc_auc > init_acc_auc:
            init_acc_auc = acc_auc
        else:
            all_cols.append(col)
    return(f'result backward selection=> best columns: {all_cols} with auc: {init_acc_auc}', all_cols)


# In[ ]:


all_cols = X.columns.to_list()
backward_col, best_cols_b = backward(X_train, y_train, X_test, y_test, all_cols)
print(backward_col)


# In[ ]:


X_train_b = pd.DataFrame(data=X_train, columns=best_cols_b)
X_test_b = pd.DataFrame(data=X_test, columns=best_cols_b)
message, yhat = LR(X_train_b, y_train, X_test_b, y_test, "backward selection")
print(message)
print(classification_report(y_test, yhat))


# In[ ]:


cm = confusion_matrix(y_test, yhat)

df1 = pd.DataFrame(columns=["0","1"], index= ["0","1"], data= cm )

f,ax = plt.subplots(figsize=(6,6))

sns.heatmap(df1, annot=True,cmap="Greens", fmt= '.0f',
            ax=ax,linewidths = 5, cbar = False,annot_kws={"size": 16})
plt.xlabel("Predicted Label")
plt.xticks(size = 12)
plt.yticks(size = 12, rotation = 0)
plt.ylabel("True Label")
plt.title("Confusion Matrix", size = 12)
plt.show()


# <a id="PCA"></a>
# # PCA

# In[ ]:


# forward selection has 6 features
pca = PCA(n_components=6)
pca.fit(X_train)
pca_train = pd.DataFrame()
pca_test = pd.DataFrame()
pca_train = pd.DataFrame(
    data = pca.transform(X_train), 
    columns =[
    "pca"+str(6) for i in range(1, 6+1)
])
pca_test = pd.DataFrame(
    data = pca.transform(X_test), 
    columns =[
    "pca"+str(6) for i in range(1, 6+1)
])


# In[ ]:


pca_train


# In[ ]:


message, yhat = LR(pca_train, y_train, pca_test, y_test, "pca 6")
print(message)
print(classification_report(y_test, yhat))


# In[ ]:


cm = confusion_matrix(y_test, yhat)

df1 = pd.DataFrame(columns=["0","1"], index= ["0","1"], data= cm )

f,ax = plt.subplots(figsize=(6,6))

sns.heatmap(df1, annot=True,cmap="Greens", fmt= '.0f',
            ax=ax,linewidths = 5, cbar = False,annot_kws={"size": 16})
plt.xlabel("Predicted Label")
plt.xticks(size = 12)
plt.yticks(size = 12, rotation = 0)
plt.ylabel("True Label")
plt.title("Confusion Matrix", size = 12)
plt.show()


# # Feature engineering

# In[ ]:


df.battery_power.describe()


# # Binning

# In[ ]:


df_bin = df.copy()


# In[ ]:


def filter(x):
    if x >= 1500:
        return 0.9
    elif 1000 <= x < 1500:
        return 0.6
    else:
        return 0.3


# In[ ]:


df_bin.battery_power = df_bin.battery_power.apply(filter)


# In[ ]:


df_bin.battery_power.describe()


# # One hot encoding

# In[ ]:


df.head(3)


# In[ ]:


df_ohe = df.copy()


# In[ ]:


for i in df_ohe.index:
    if df_ohe.loc[i, 'blue'] == 0:
        df_ohe.loc[i, 'notblue'] = 1
    else:
        df_ohe.loc[i, 'notblue'] = 0
for i in df_ohe.index:
    if df_ohe.loc[i, 'dual_sim'] == 0:
        df_ohe.loc[i, 'notdual_sim'] = 1
    else:
        df_ohe.loc[i, 'notdual_sim'] = 0
for i in df_ohe.index:
    if df_ohe.loc[i, 'four_g'] == 0:
        df_ohe.loc[i, 'notfour_g'] = 1
    else:
        df_ohe.loc[i, 'notfour_g'] = 0
        
df_ohe.head(3)


# In[ ]:


df_ohe.drop(columns=['price_range'], axis=1, inplace=True)
df_ohe['price_range'] = df.price_range
df_ohe.head(3)


# # Transofrm

# In[ ]:


# Visualizing the distribution for every "feature"
df.hist(edgecolor="black", linewidth=1.2, figsize=(20, 20))
plt.show()


# In[ ]:


df_tra = df.copy()


# In[ ]:


def filter(x):
    if x > 0:
        return np.log(x)
    else:
        return x


# In[ ]:


df_tra.sc_w = df_tra.sc_w.apply(filter)
df_tra.fc = df_tra.fc.apply(filter)


# In[ ]:


# Visualizing the distribution for every "feature"
df_tra.hist(edgecolor="black", linewidth=1.2, figsize=(20, 20))
plt.show()


# # Area

# In[ ]:


df_area = df.copy()


# In[ ]:


df_area['sc_area'] = df['sc_h'] * df['sc_w']
df_area['px_area'] = df['px_height'] * df['px_width']
df_area.drop(columns=['price_range'], axis=1, inplace=True)
df_area['price_range'] = df.price_range


# In[ ]:


df_area.head()


# # ALL

# In[ ]:


df_all = df.copy()


# In[ ]:


def filter(x):
    if x >= 1500:
        return 0.9
    elif 1000 <= x < 1500:
        return 0.6
    else:
        return 0.3


# In[ ]:


df_all.battery_power = df_all.battery_power.apply(filter)


# In[ ]:


def filter(x):
    if x > 0:
        return np.log(x)
    else:
        return x


# In[ ]:


df_all.sc_w = df_all.sc_w.apply(filter)
df_all.fc = df_all.fc.apply(filter)


# In[ ]:


for i in df_all.index:
    if df_all.loc[i, 'blue'] == 0:
        df_all.loc[i, 'notblue'] = 1
    else:
        df_all.loc[i, 'notblue'] = 0


# In[ ]:


df_all['sc_area'] = df['sc_h'] * df['sc_w']
df_all['px_area'] = df['px_height'] * df['px_width']
df_all.drop(columns=['price_range'], axis=1, inplace=True)
df_all['price_range'] = df.price_range


# # Compare thease data

# In[ ]:


for i in [[df_bin, 'bininng'], [df_ohe, 'ohe hot encoding'], [df_tra, 'log transform'], [df_area, 'area'],[df_all, 'all']]:
    data = i[0].copy()
    y = data.pop('price_range')
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    fit_X = scaler.fit(X_train)
    X_train = fit_X.transform(X_train)
    X_test = fit_X.transform(X_test)
    X_train = pd.DataFrame(data=X_train, columns=X.columns)
    X_test = pd.DataFrame(data=X_test, columns=X.columns)
    message, yhat = LR(X_train, y_train, X_test, y_test, i[1])
    print(message)


# # Crossvalidation, Bootstrapping

# **Crossvalidation**
# > Cross-Validation is a statistical method of evaluating and comparing learning algorithms by dividing data into two segments: one used to learn or train a model and the other used to validate the model.
# 
# **Bootstrapping**
# > Bootstrapping is a statistical procedure that resamples a single dataset to create many simulated samples.
# 
# **Why Crossvalidation**
# > It is used to protect against overfitting in a predictive model, particularly in a case where the amount of data may be limited.<br>
# Advantages of cross-validation: More accurate estimate of out-of-sample accuracy. More “efficient” use of data as every observation is used for both training and testing.
# 
# **Why Bootstrapping**
# > It helps in avoiding overfitting and improves the stability of machine learning algorithms.<br>
# A great advantage of bootstrap is its simplicity. It is a straightforward way to derive estimates of standard errors and confidence intervals for complex estimators of the distribution, such as percentile points, proportions, odds ratio, and correlation coefficients.
# 
# **When Bootstrapping, Crossvalidation**
# > We use Bootsrap when we one to estimate parameteters.<br>
# We use Crossvalodation for our model's score.
# 
# **5x2 cross validation**
# > 5x2cv refer to a 5 repetition of a 2-fold. do a 2-fold (50/50 split between train and test), repeat it 4 more times. The 5x2cv was for comparing supervised classification learning algorithms by Dietterich as a way of obtaining not only a good estimate of the generalisation error but also a good estimate of the variance of that error (in order to perform statistical tests). [code on kaggle](https://www.kaggle.com/code/ogrellier/parameter-tuning-5-x-2-fold-cv-statistical-test/notebook) 

# # Decision-tree, Pruning

# **CART, ID3**
# > CART is a classification algorithm for building a decision tree based on Gini's impurity index as splitting criterion.<br>
#  ID3 is a classification algorithm for building a decision tree based on Information Gain as splitting criterion. <br>
# 
# *Another difference:*<br>
# The CART algorithm produces only binary Trees: non-leaf nodes always have two children. <br>
# ID3 can produce Decision Trees with nodes having more than two children.

# In[ ]:


tree = DecisionTreeClassifier(criterion='gini', ccp_alpha=0, max_leaf_nodes=8, min_samples_split=4, max_depth=11)


# In[ ]:


tree.fit(X_train, y_train)


# In[ ]:


y_pred_tree = tree.predict(X_test)


# In[ ]:


print('DecisionTreeRegressor: test', tree.score(X_train, y_train))
print('DecisionTreeRegressor R^2: test', metrics.r2_score(y_test, y_pred_tree))


# In[ ]:


tree = DecisionTreeClassifier(criterion='entropy', ccp_alpha=0, max_depth=11)


# In[ ]:


tree.fit(X_train, y_train)


# In[ ]:


y_pred_tree = tree.predict(X_test)


# In[ ]:


print('DecisionTreeRegressor: test', tree.score(X_train, y_train))
print('DecisionTreeRegressor R^2: test', metrics.r2_score(y_test, y_pred_tree))


# **Pruning**
# > Pruning is a data compression technique in machine learning and search algorithms that reduces the size of decision trees by removing sections of the tree that are non-critical and redundant to classify instances.<br>
# > We use it because it can prevent overfitting<br>
#     max_leaf_nodes. Reduce the number of leaf nodes.<br>
#     min_samples_leaf. Restrict the size of sample leaf. Minimum sample size in terminal nodes can be fixed to 30, 100, 300 or 5% of total.<br>
#     max_depth. Reduce the depth of the tree to build a generalized tree.<br>

# In[ ]:


tree = DecisionTreeClassifier(criterion='entropy', ccp_alpha=0, max_depth=8, max_leaf_nodes=10, min_samples_leaf=30)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
print('DecisionTreeRegressor: test', tree.score(X_train, y_train))
print('DecisionTreeRegressor R^2: test', metrics.r2_score(y_test, y_pred_tree))


# **Elbow method**
# > We are looking for a trade of between bias and variance 

# **Matthews Correlation Coefficient(MCC)**
# >The MCC is in essence a correlation coefficient value between -1 and +1. A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction. The statistic is also known as the phi coefficient.<br>
# MCC = (TP*TN – FP*FN) / √(TP+FP)(TP+FN)(TN+FP)(TN+FN)

# # Random forest

# In[ ]:


random = RandomForestClassifier()


# In[ ]:


random.fit(X_train, y_train)


# In[ ]:


y_pred_random = random.predict(X_test)


# In[ ]:


print('RandomForestClassifier: test', random.score(X_train, y_train))
print('RandomForestClassifier R^2: test', metrics.r2_score(y_test, y_pred_random))


# # Bagging

# In[ ]:


bagging = BaggingClassifier()


# In[ ]:


bagging.fit(X_train, y_train)


# In[ ]:


y_pred_bagging = bagging.predict(X_test)


# In[ ]:


print('BaggingClassifier: test', bagging.score(X_train, y_train))
print('BaggingClassifier R^2: test', metrics.r2_score(y_test, y_pred_bagging))


# # Boosting

# In[ ]:


boosting = GradientBoostingClassifier(ccp_alpha=0, learning_rate=0.08)


# In[ ]:


boosting.fit(X_train, y_train)


# In[ ]:


y_pred_boosting = boosting.predict(X_test)


# In[ ]:


print('GradientBoostingClassifier: test', boosting.score(X_train, y_train))
print('GradientBoostingClassifier R^2: test', metrics.r2_score(y_test, y_pred_boosting))


# # KNN

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=10)


# In[ ]:


knn.fit(X_train, y_train)


# In[ ]:


y_pred_knn = knn.predict(X_test)


# In[ ]:


print('KNeighborsClassifier: test', knn.score(X_train, y_train))
print('KNeighborsClassifier R^2: test', metrics.r2_score(y_test, y_pred_knn))


# # Predict

# In[ ]:


df_org_test = pd.read_csv("/kaggle/input/mobile-price-classification/test.csv", index_col=['id'])
df_test = df_org_test.copy()


# In[ ]:


df.columns


# In[ ]:


df_test.columns


# In[ ]:


X_train_p = fitter_1.transform(df_test)


# In[ ]:


X_train_p = pd.DataFrame(data=X_train_p, columns=df_test.columns)


# In[ ]:


X_train_p


# In[ ]:


# Use backward columns
cols = ['wifi', 'battery_power', 'blue', 'clock_speed', 'dual_sim',
        'fc', 'four_g','int_memory', 'm_dep', 'mobile_wt', 'n_cores',
        'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w','talk_time',
        'three_g']


# In[ ]:


X_train_p = X_train_p[cols]
X_train_p


# In[ ]:


acc_auc = LR(X_train_b, y_train, X_test_b, y_test, i=22)


# In[ ]:


acc_auc


# In[ ]:


LR = LR(X_train_b, y_train, X_test_b, y_test, i=23)


# In[ ]:


LR.predict(X_train_p)

