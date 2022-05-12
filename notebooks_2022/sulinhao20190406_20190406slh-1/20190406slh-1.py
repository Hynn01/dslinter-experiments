#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 引入csv文件读取库
import pandas as pd
# 引入画图库
import matplotlib.pyplot as plt
#import seaborn as sns
# 引入数学运算库
import numpy as np
from sklearn.model_selection import GridSearchCV
# 引入数据处理包
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler  # 对数据做归一化
from sklearn.ensemble import RandomForestClassifier # 随机森林模型
from  sklearn.tree import DecisionTreeClassifier # 决策树模型
from sklearn.neural_network import MLPClassifier as DNN
from sklearn.metrics import f1_score # f1_score评价指标
from sklearn.model_selection import KFold, cross_val_score  # K折交叉验证
#from imblearn.over_sampling import RandomOverSampler, SMOTE # 重采样解决样本不平衡
import scipy.stats as st # 数据分析包

# 重新读一下数据
df_train = pd.read_csv('Dataset.csv')
df_test = pd.read_csv('Testing.csv')
# 把ID去掉，没有作用
test_ID = df_test['ID'] # 后面用来包装csv
df_train = df_train.drop(columns='ID')
df_test = df_test.drop(columns='ID')

# 去掉3号和9号
df_train = df_train.drop(index = df_train[df_train['quality'] == 3].index)
df_train = df_train.drop(index = df_train[df_train['quality'] == 9].index)
df_train = df_train.reset_index(drop=True)

# 下面的内容和上面一样 ----------------------------

# 把quality拿出来，因为是我们要预测的东西
quality = df_train['quality']
df_train = df_train.drop(columns='quality')

# 下面对所有数值型数据做归一化，就ok了
min_max_scaler = MinMaxScaler()
# 这里直接对训练集和测试集都做了
min_max_scaler.fit(pd.concat([df_train,df_test]).values)
all_data = min_max_scaler.transform(pd.concat([df_train,df_test]).values)

# 把train和test的分出来, 到此，数据处理就结束了，这些数据已经可以送到模型中训练了
train_data = all_data[:len(df_train)]
test_data = all_data[len(df_train):]
#预测
# 其实里面有不少参数，但这里就用默认的好了

temp_csv = pd.DataFrame()
n_splits = 5
mean_f1_score = 0
kf = KFold(n_splits=n_splits, shuffle=True) # 使用sklearn自带划分k折的函数
#scorel=[]
#for i in range(165,175):
#    model=RandomForestClassifier(n_estimators=i,n_jobs=-1,random_state=90)
#    score=cross_val_score(model,train_data,quality,cv=10).mean()
#    scorel.append(score)
#print(max(scorel),(scorel.index(max(scorel))+165))
#plt.figure(figsize=[20,5])
#plt.plot(range(165,175),scorel)
#plt.show()
#param_grid={'max_features':range(3,11,1)}
#model=RandomForestClassifier(n_estimators=173,random_state=90,criterion="gini",max_depth=3)
#GS=GridSearchCV(model,param_grid,cv=10)
#GS.fit(train_data,quality)
#print(GS.best_score_)
#print(GS.best_params_)

#dnn = DNN(hidden_layer_sizes=(100,),max_iter=500,random_state=420)

for fold, (trn_idx, val_idx) in enumerate(kf.split(train_data, quality)):
    print('fold:', fold)
    X_train, y_train = train_data[trn_idx], quality[trn_idx]
    X_val, y_val = train_data[val_idx], quality[val_idx]
    # 定义模型
    #model=AdaBoostClassifier(DecisionTreeClassifier(max_depth=8,min_samples_split=20,min_samples_leaf=5),n_estimators=100,algorithm="SAMME",learning_rate=0.8)
    model=RandomForestClassifier(random_state=0)
    #model=AdaBoostClassifier(DNN(hidden_layer_sizes=(200,),max_iter=500,random_state=420))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    # 调用skelarn计算f1 score的函数
    f1 = f1_score(y_val, y_pred, average='macro')
    mean_f1_score += f1
    print('f1_socre is :', f1)
   # 用这一轮的模型，去预测test
    test_pred = model.predict(test_data)
    pred_name = str(fold)+'_'+'quality'
    temp_csv[pred_name] = test_pred
print('mean f1_score is :', mean_f1_score / n_splits)
sbu_csv = pd.DataFrame()
sbu_csv['ID'] = test_ID
# 取所有预测的众数
sbu_csv['quality'] = temp_csv.mode(axis=1)[0].map(lambda x:int(x))  # 取众数是float型，改为int型
sbu_csv.to_csv('submission.csv', index=False)# This Python 3 environment comes with many helpful analytics libraries installed
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

