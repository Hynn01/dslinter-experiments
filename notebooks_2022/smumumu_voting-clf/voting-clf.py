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


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.metrics import classification_report,accuracy_score

from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import fowlkes_mallows_score, silhouette_score, classification_report, mean_absolute_error,     mean_squared_error, median_absolute_error, explained_variance_score, r2_score
from sklearn.preprocessing import StandardScaler#标准差标准化
from sklearn.ensemble import VotingClassifier      #投票分类器
from sklearn.ensemble import ExtraTreesClassifier


# In[ ]:


train_data = pd.read_csv("../input/bigdata-and-datamining-1st-ex/Dataset.csv")


# In[ ]:


x = train_data.iloc[:,1:12].values 
y = train_data['quality'].values
ros = RandomOverSampler()
x,y = ros.fit_resample(x,y)
print(pd.DataFrame(y)[0].value_counts().sort_index())


# In[ ]:


train_features = train_data.iloc[:,1:12]
train_labels = train_data.iloc[:,12]


# In[ ]:


Xtrain, Xtest, Ytrain, Ytest = train_test_split(train_features,train_labels,test_size=0.1)


# In[ ]:


stdScaler = StandardScaler().fit(Xtrain)#stdScaler存有计算出来的均值和方差
std_Xtrain = stdScaler.transform(Xtrain)#使用stdScaler中的均值和方差使得data_train归一化
std_Xtest = stdScaler.transform(Xtest)


# In[ ]:


erfc = ExtraTreesClassifier()
erfc = erfc.fit(std_Xtrain, Ytrain)
target_test = erfc.predict(std_Xtest)#结果预测
true = np.sum(target_test == Ytest)
accuracy=true/Ytest.shape[0]
print("erfc算法：")
print('预测正确结果：',true)
print('预测错误结果：',Ytest.shape[0]-true)
print('正确率：',accuracy)


# In[ ]:


rfc = RandomForestClassifier()
rfc = rfc.fit(std_Xtrain, Ytrain)
target_test = rfc.predict(std_Xtest)#结果预测
true = np.sum(target_test == Ytest)
accuracy=true/Ytest.shape[0]
print("rfc算法：")
print('预测正确结果：',true)
print('预测错误结果：',Ytest.shape[0]-true)
print('正确率：',accuracy)


# In[ ]:


voting_clf = VotingClassifier(estimators=[('erfc',erfc),('rfc',rfc)],#estimators:子分类器
                              voting='soft') #参数voting代表你的投票方式，hard,soft
voting_clf.fit(std_Xtrain,Ytrain)
target_test = voting_clf.predict(std_Xtest)#结果预测
true = np.sum(target_test == Ytest)
accuracy=true/Ytest.shape[0]
print("voting_clf算法：")
print('预测正确结果：',true)
print('预测错误结果：',Ytest.shape[0]-true)
print('正确率：',accuracy)


# In[ ]:


test_data = pd.read_csv("../input/bigdata-and-datamining-1st-ex/Testing.csv")


# In[ ]:


test = test_data.iloc[:,1:12]
stdScaler = StandardScaler().fit(test)
std_test = stdScaler.transform(test)


# In[ ]:


predict = voting_clf.predict(std_test)
predict


# In[ ]:


test_data['quality']=pd.Series(predict.reshape(1, -1)[0])
output = pd.DataFrame({'ID': test_data.ID, 'quality': test_data.quality})
output.to_csv('submission.csv', index=False)

