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




import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier  # 导入sklearn库的RandomForestClassifier



#导入数据集
train = pd.read_csv('/kaggle/input/bigdata-and-datamining-1st-ex/Dataset.csv',sep=',',header=0) 
test = pd.read_csv('/kaggle/input/bigdata-and-datamining-1st-ex/Testing.csv',sep=',',header=0) 

#设置x_train，y_train
x_train = train.drop(["ID","quality"],axis=1).values  #drop删除ID和quality两列
y_train = train["quality"].values
x_test = test.drop(["ID"],axis=1).values
test_ID = test['ID']  #用于最终要求的csv文件
x_train_scale = preprocessing.scale(x_train)
x_test_scale = preprocessing.scale(test[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']])




# 随机森林模型
model = RandomForestClassifier(n_estimators=200) # 实例化模型RandomForestClassifier
model.fit(x_train_scale, y_train)  # 在训练集上训练模型
#score = model.score(x_train_scaled,y_train)  #训练好的模型在训练集上进行评分，返回精确度
#print(score)
score = cross_val_score(model,x_train_scale,y_train,cv=8)

#可以利用GridSearchCV，ridSearch和CV，即网格搜索和交叉验证。
#网格搜索，搜索的是参数，在指定的参数范围内，按步长依次调整参数，利用调整的参数训练学习器，从所有的参数中找到在验证集上精度最高的参数，这其实是一个训练和比较的过程。
#k折交叉验证将所有数据集分成k份，不重复地每次取其中一份做测试集，用其余k-1份做训练集训练模型，之后计算该模型在测试集上的得分,将k次的得分取平均得到最后的得分。

#利用建立的随机森林模型进行预测
clf = RandomForestClassifier(criterion='entropy',max_features=2,n_estimators=320)
clf = clf.fit(x_train_scale, y_train)
predict = clf.predict(x_test_scale)

#以csv形式存储结果
sub_csv = pd.DataFrame()
sub_csv['ID'] = test_ID
sub_csv['quality'] = predict
sub_csv.to_csv('submission.csv', index=False)

