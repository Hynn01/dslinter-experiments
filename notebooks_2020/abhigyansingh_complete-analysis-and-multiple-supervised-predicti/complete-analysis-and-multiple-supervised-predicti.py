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


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


master = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")


# In[ ]:


def check_missing_data(df):
    flag=df.isna().sum().any()
    if flag==True:
        total = df.isnull().sum()
        percent = ((df.isnull().sum())/len(df))*100
        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        data_type = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            data_type.append(dtype)
        output['Types'] = data_type
        return(np.transpose(output))
    else:
        return(False)


# In[ ]:


check_missing_data(master)


# In[ ]:


master.shape


# In[ ]:


master.info()


# In[ ]:


master.head(5)


# In[ ]:


master.drop('sl_no',axis=1,inplace = True)


# In[ ]:


cat = master.drop('status',axis=1).select_dtypes(include = object).columns
for i in cat:
    print(i)
    print(master[i].value_counts())
    print( )
    print( )


# ## Let's visualize placement among the gender

# In[ ]:


plt.figure(figsize = (10,8))
sns.countplot('gender',data =master)


# In[ ]:


plt.figure(figsize = (10,8))
sns.countplot("gender",data= master,hue='status')


# In[ ]:


print("out of {} Boys,".format(len(master[master['gender']=='M'])),"{} got placed,".format(len(master[(master['gender']=='M') & (master['status']=='Placed')])),"which is {} % of placement among males.".format(len(master[(master['gender']=='M') & (master['status']=='Placed')])/len(master[master['gender']=='M']) * 100))


# In[ ]:


print("out of {} girls,".format(len(master[master['gender']=='F'])),"{} got placed,".format(len(master[(master['gender']=='F') & (master['status']=='Placed')])),"which is {} % of placement among females.".format(len(master[(master['gender']=='F') & (master['status']=='Placed')])/len(master[master['gender']=='F']) * 100))


# ## Let's see people belonging to which board in 10th got placed.

# In[ ]:


plt.figure(figsize = (10,8))
sns.countplot("ssc_b",data = master,hue='status')


# People belonging to the central board got placed the most however there is not much of a difference.

# ## Let's see about 12th board

# In[ ]:


plt.figure(figsize = (10,8))
sns.countplot('hsc_b',data = master,hue = 'status')


# Turns out people in other boards in 12th got placed,maybe because they got good marks.
# Let's see how many of them moved from central in 10th to other boards for 12th.

# In[ ]:


print(len(master[(master['ssc_b']=='Central') & (master['hsc_b']=='Others')])," people studied in central board for their 10th and moved to some other board for their 12th")
print( )
print(len(master[(master['ssc_b']=='Central') & (master['hsc_b']=='Central')])," people stayed in Cetral for both their 10th and 12th board. ")
print ( )
print(len(master[(master['ssc_b']=='Others') & (master['hsc_b']=='Central')])," people studied in other boards for their 10th and moved to Central board for their 12th")
print( )
print(len(master[(master['ssc_b']=='Others') & (master['hsc_b']=='Others')])," people never came to Central board")


# ## Let's see the highest placement happened for people having which degree.

# In[ ]:


plt.figure(figsize = (10,8))
sns.countplot('degree_t',data = master,hue = 'status')


# Humongous amount of people that were placed belonged to commerce and management.

# ## Does work experience had an impact on students who were placed?

# In[ ]:


plt.figure(figsize = (10,8))
sns.countplot('workex',data = master,hue = 'status')


# Seems that experience was not taken into account for placement.

# Let's see how each categorical variable is significant to the target label.For that i will use the Cramer's V test

# In[ ]:


# Cramer's v tests
def cramers_stat(confusion_matrix):
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    return np.sqrt(chi2 / (n*(min(confusion_matrix.shape)-1)))

for i in cat:
  ctest = pd.crosstab(master[i],master['status'])
  print(i,"test with target data")
  print( )
  (chi2,p,dof,_) = stats.chi2_contingency([ctest.iloc[0].values,ctest.iloc[1].values])
  print("Chi-Squared :",chi2)
  print("P-value :",p)
  print("Freedom :",dof)
  print("Cramer's test :",cramers_stat(ctest))
  print( )
  print( )


# Only work experience and specialization were the categorical columns that had an impact to our target variable.We can remove the rest.

# Let's see the correlation between features in the data

# In[ ]:


plt.figure(figsize = (10,8))
sns.heatmap(master.corr(method = 'spearman'),annot = True)


# No variables show high correlation to each other.

# In[ ]:


sns.pairplot(master,diag_kind = 'kde',hue = 'status',dropna = True)


# In[ ]:


master.select_dtypes(exclude = object).columns


# In[ ]:


plt.figure(figsize = (10,8))
sns.scatterplot('salary','etest_p',data = master)


# people with higher employbility test does not guarantee higher salary

# In[ ]:


plt.figure(figsize = (10,8))
sns.scatterplot('degree_p','salary',data = master)


# higher percentage in college does not guarantee higher package.

# # missing value imputation

# Let's impute the missing salary to 0 as those missing data due to people not getting placed.

# In[ ]:


master[master.isnull().any(axis = 1)]


# creating a copy of master data so that if anything goes wrong i can come back to my main master dataframe.

# In[ ]:


data = master.copy()


# In[ ]:


data = data.fillna(0)


# In[ ]:


check_missing_data(data)


# # Feature selection

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


data_enc = data.apply(LabelEncoder().fit_transform)


# In[ ]:


x = data_enc.drop('status',axis = 1)
y = data_enc['status']


# In[ ]:


rnd = RandomForestClassifier().fit(x,y)


# In[ ]:


feature_importances = pd.DataFrame((rnd.feature_importances_*100),index = x.columns,columns=['importance']).sort_values('importance',ascending=False)


# In[ ]:


feature_importances


# In[ ]:


plt.figure(figsize = (20,4))
plt.plot(feature_importances)


# If you go up and check the cramer's v test ,we already found out that the which categorical variables that a significance to our target variable.and here it's confirmed,So let's just drop the variable after etest_p

# In[ ]:


index_drop = feature_importances[feature_importances['importance']<1.5].index


# In[ ]:


data_imp = data_enc.drop(index_drop,axis = 1)


# In[ ]:


data_imp.head()


# # Different Supervised Learning

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,f1_score


# In[ ]:


X = data_imp.drop('status',axis = 1)
Y = data_imp['status']


# In[ ]:


data_imp.shape


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3,random_state = 123)


# # LogisticRegression

# In[ ]:


Log_model = LogisticRegression().fit(x_train,y_train)


# In[ ]:


Logpred = Log_model.predict(x_test)


# In[ ]:


print ("Confusion Matrix :", pd.crosstab(y_test,Logpred))
print( )
print("Accuracy :",accuracy_score(y_test,Logpred))
print( )
print("F1-Score :",f1_score(y_test,Logpred))
print( )
print("auc_roc_score :", roc_auc_score(y_test,Logpred))
print( )
fpr,tpr,thres = roc_curve(y_test,Logpred)
plt.figure(figsize = (10,8))
plt.plot(fpr,tpr)
plt.plot([0,1],ls = '--')
plt.plot([0,0],[1,0],c = "0.7")
plt.plot([1,1],c = "0.7")
plt.xlabel("FalsePositiveRate")
plt.ylabel("TruePositiveRate")
plt.show()


# # DecisionTree

# In[ ]:


tree_model = DecisionTreeClassifier().fit(x_train,y_train)


# In[ ]:


treepred = tree_model.predict(x_test)


# In[ ]:


print ("Confusion Matrix :", pd.crosstab(y_test,treepred))
print( )
print("Accuracy :",accuracy_score(y_test,treepred))
print( )
print("F1-Score :", f1_score(y_test,treepred))
print( )
print("auc_roc_score :", roc_auc_score(y_test,treepred))
print( )
fpr,tpr,thres = roc_curve(y_test,treepred)
plt.figure(figsize = (10,8))
plt.plot(fpr,tpr)
plt.plot([0,1],ls = '--')
plt.plot([0,0],[1,0],c = "0.7")
plt.plot([1,1],c = "0.7")
plt.xlabel("FalsePositiveRate")
plt.ylabel("TruePositiveRate")
plt.show()


# # RandomForestClassifier

# In[ ]:


rnd_model = RandomForestClassifier().fit(x_train,y_train)


# In[ ]:


rndpred = rnd_model.predict(x_test)


# In[ ]:


print ("Confusion Matrix :", pd.crosstab(y_test,rndpred))
print( )
print("Accuracy :",accuracy_score(y_test,rndpred))
print( )
print("F1-Score :", f1_score(y_test,rndpred))
print( )
print("auc_roc_score :", roc_auc_score(y_test,rndpred))
print( )
fpr,tpr,thres = roc_curve(y_test,rndpred)
plt.figure(figsize = (10,8))
plt.plot(fpr,tpr)
plt.plot([0,1],ls = '--')
plt.plot([0,0],[1,0],c = "0.7")
plt.plot([1,1],c = "0.7")
plt.xlabel("FalsePositiveRate")
plt.ylabel("TruePositiveRate")
plt.show()


# # ExtraTreeClassifier

# In[ ]:


etr_model = ExtraTreesClassifier().fit(x_train,y_train)


# In[ ]:


etrpred = etr_model.predict(x_test)


# In[ ]:


print ("Confusion Matrix :", pd.crosstab(y_test,etrpred))
print( )
print("Accuracy :",accuracy_score(y_test,etrpred))
print( )
print("F1-Score :", f1_score(y_test,etrpred))
print( )
print("auc_roc_score :", roc_auc_score(y_test,etrpred))
print( )
fpr,tpr,thres = roc_curve(y_test,etrpred)
plt.figure(figsize = (10,8))
plt.plot(fpr,tpr)
plt.plot([0,1],ls = '--')
plt.plot([0,0],[1,0],c = "0.7")
plt.plot([1,1],c = "0.7")
plt.xlabel("FalsePositiveRate")
plt.ylabel("TruePositiveRate")
plt.show()


# # XGBoost

# In[ ]:


xgb_model = XGBClassifier().fit(x_train,y_train)


# In[ ]:


xgbpred = xgb_model.predict (x_test)


# In[ ]:


print ("Confusion Matrix :", pd.crosstab(y_test,xgbpred))
print( )
print("Accuracy :",accuracy_score(y_test,xgbpred))
print( )
print("F1-Score :", f1_score(y_test,xgbpred))
print( )
print("auc_roc_score :", roc_auc_score(y_test,xgbpred))
print( )
fpr,tpr,thres = roc_curve(y_test,xgbpred)
plt.figure(figsize = (10,8))
plt.plot(fpr,tpr)
plt.plot([0,1],ls = '--')
plt.plot([0,0],[1,0],c = "0.7")
plt.plot([1,1],c = "0.7")
plt.xlabel("FalsePositiveRate")
plt.ylabel("TruePositiveRate")
plt.show()


# # KNN

# In[ ]:


knn_model = KNeighborsClassifier().fit(x_train,y_train)


# In[ ]:


knnpred = knn_model.predict(x_test)


# In[ ]:


print ("Confusion Matrix :", pd.crosstab(y_test,knnpred))
print( )
print("Accuracy :",accuracy_score(y_test,knnpred))
print( )
print("F1-Score :", f1_score(y_test,knnpred))
print( )
print("auc_roc_score :", roc_auc_score(y_test,knnpred))
print( )
fpr,tpr,thres = roc_curve(y_test,knnpred)
plt.figure(figsize = (10,8))
plt.plot(fpr,tpr)
plt.plot([0,1],ls = '--')
plt.plot([0,0],[1,0],c = "0.7")
plt.plot([1,1],c = "0.7")
plt.xlabel("FalsePositiveRate")
plt.ylabel("TruePositiveRate")
plt.show()


# # SVM

# In[ ]:


svm_model = svm.SVC().fit(x_train,y_train)


# In[ ]:


svmpred = svm_model.predict(x_test)


# In[ ]:


print ("Confusion Matrix :", pd.crosstab(y_test,svmpred))
print( )
print("Accuracy :",accuracy_score(y_test,svmpred))
print( )
print("F1-Score :", f1_score(y_test,svmpred))
print( )
print("auc_roc_score :", roc_auc_score(y_test,svmpred))
print( )
fpr,tpr,thres = roc_curve(y_test,svmpred)
plt.figure(figsize = (10,8))
plt.plot(fpr,tpr)
plt.plot([0,1],ls = '--')
plt.plot([0,0],[1,0],c = "0.7")
plt.plot([1,1],c = "0.7")
plt.xlabel("FalsePositiveRate")
plt.ylabel("TruePositiveRate")
plt.show()

