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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso,Ridge,ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# **導入資料集**

# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# **查看資料**

# In[ ]:


train.head()


# **總共有80個特徵用來預測SalePrice** 

# In[ ]:


test.head()


# In[ ]:


train.info()


# In[ ]:


test.info()


# 看來有很多資料有缺失，下面計算各欄位的缺失值個數。

# # *Missing Value*

# In[ ]:


data_null_number=pd.DataFrame(train.isnull().sum().sort_values(ascending = False),columns=['Null number'])
data_null_number.head(20)


# 我要把缺失超過100個以上的特徵丟掉

# In[ ]:


#drop missing value >100 & ID
train = train.drop(data_null_number[data_null_number['Null number']>100].index,axis=1)
test = test.drop(data_null_number[data_null_number['Null number']>100].index,axis=1)


# 接下來查看有相同缺失數量81的特徵,GarageType,GarageCond,GarageYrBlt,GarageFinish,GarageQual  
# Garage...指的是車庫的...，可見這五個特徵是一組的。

# In[ ]:


Garage_data=train[['GarageType','GarageCond','GarageYrBlt','GarageFinish','GarageQual']]
print(Garage_data['GarageType'].unique())
print(Garage_data['GarageCond'].unique())
print(Garage_data['GarageYrBlt'].unique())
print(Garage_data['GarageFinish'].unique())
print(Garage_data['GarageQual'].unique())


# 'GarageType','GarageCond','GarageFinish','GarageQual'是類別資料，來看看裡面有什麼類別。

# In[ ]:


is_NaN = Garage_data.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = Garage_data[row_has_NaN]
rows_with_NaN.head(50),rows_with_NaN.tail(31)


# 好像資料中出現的缺失值是沒有的意思，我想將nan轉變成No_Garage。

# In[ ]:


Garage_data=Garage_data.fillna(value='No_data')


# In[ ]:


#放回到train裡
train[['GarageType','GarageCond','GarageYrBlt','GarageFinish','GarageQual']]=Garage_data
print(train[['GarageType','GarageCond','GarageYrBlt','GarageFinish','GarageQual']].isnull().sum())


# BsmtFinType2,BsmtExposure,BsmtQual,BsmtCond,BsmtFinType1看起來也是同組的資料。  
# 只是前兩個特徵多了一個缺失。我來檢查看看。

# In[ ]:


Bsmt_data=train[['BsmtFinType2','BsmtExposure','BsmtQual','BsmtCond','BsmtFinType1']]
print('BsmtFinType2:',Bsmt_data['BsmtFinType2'].unique())
print('BsmtExposure:',Bsmt_data['BsmtExposure'].unique())
print('BsmtQual:',Bsmt_data['BsmtQual'].unique())
print('BsmtCond:',Bsmt_data['BsmtCond'].unique())
print('BsmtFinType1:',Bsmt_data['BsmtFinType1'].unique())


# "BsmtExposure"有"No"的類別，這下我不知該特徵的nan是不是"No"的意思。  
# 看看這幾個特徵缺失資料長怎樣

# In[ ]:


is_NaN = Bsmt_data.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = Bsmt_data[row_has_NaN]
rows_with_NaN


# 我認為"BsmtExposure"的nan是缺失值(row=948)，因為BsmtQual,BsmtCond,BsmtFinType1都有值，在No的row也是。所以等等找出最有可能的值填入。  
# BsmtFinType2的nan就是No的意思，表示不存在。

# In[ ]:


#找出最符合的可能
BsmtExposure_count=Bsmt_data.loc[(Bsmt_data['BsmtFinType1']=='Unf')&(Bsmt_data['BsmtCond']=='TA')&
                                 (Bsmt_data['BsmtQual']=='Gd')&(Bsmt_data['BsmtFinType2']=='Unf')]['BsmtExposure'].value_counts()
print(f'結論:\nBsmtExposure_Nan最有可能的值是{BsmtExposure_count.index[BsmtExposure_count.argmax()]}')


# In[ ]:


#取代nan
Bsmt_data[['BsmtExposure']].loc[948,:].fillna(value='No')
Bsmt_data=Bsmt_data.fillna(value="No_data")
#替換train
train[['BsmtFinType2','BsmtExposure','BsmtQual','BsmtCond','BsmtFinType1']]=Bsmt_data
print(train[['BsmtFinType2','BsmtExposure','BsmtQual','BsmtCond','BsmtFinType1']].isnull().sum())


# MasVnrArea,MasVnrType,Electrical我直接填no_data

# In[ ]:


MasVnr_data=train[['MasVnrArea','MasVnrType','Electrical']]
MasVnr_data=MasVnr_data.fillna(value="No_data")
train[['MasVnrArea','MasVnrType','Electrical']]=MasVnr_data
print(train.isnull().sum())


# ok! now train is no missing value. 

# In[ ]:


data_null_number=pd.DataFrame(test.isnull().sum().sort_values(ascending = False),columns=['Null number'])
data_null_number.head(20)


# GarageYrBlt,GarageFinish,GarageQual,GarageCond,GarageType跟train遇到的一樣。

# In[ ]:


Garage_data=test[['GarageYrBlt','GarageFinish','GarageQual','GarageCond','GarageType']]
print(Garage_data['GarageYrBlt'].unique())
print(Garage_data['GarageFinish'].unique())
print(Garage_data['GarageQual'].unique())
print(Garage_data['GarageCond'].unique())
print(Garage_data['GarageType'].unique())


# In[ ]:


is_NaN = Garage_data.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = Garage_data[row_has_NaN]
rows_with_NaN.head(50),rows_with_NaN.tail(28)


# row(666)&row(1116)的GarageType有'Detchd'，但都沒有建造年分。  
# 我想這是誤植的資料，應該要把它刪除。

# In[ ]:


Garage_data.loc[666,'GarageType']='No_data'
Garage_data.loc[1116,'GarageType']='No_data'
Garage_data=Garage_data.fillna(value='No_data')
test[['GarageYrBlt','GarageFinish','GarageQual','GarageCond','GarageType']]=Garage_data
print(test[['GarageYrBlt','GarageFinish','GarageQual','GarageCond','GarageType']].isnull().sum())


# In[ ]:


Bsmt_data=test[['BsmtFinType2','BsmtExposure','BsmtQual','BsmtCond','BsmtFinType1']]
print('BsmtFinType2:',Bsmt_data['BsmtFinType2'].unique())
print('BsmtExposure:',Bsmt_data['BsmtExposure'].unique())
print('BsmtQual:',Bsmt_data['BsmtQual'].unique())
print('BsmtCond:',Bsmt_data['BsmtCond'].unique())
print('BsmtFinType1:',Bsmt_data['BsmtFinType1'].unique())


# In[ ]:


is_NaN = Bsmt_data.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = Bsmt_data[row_has_NaN]
rows_with_NaN


# row[27,580,725,757,758,888,1064]應該是要有值

# In[ ]:


for i in [27,580,725,757,758,888,1064]:
    print(Bsmt_data.loc[i,:])


# In[ ]:


#找出最符合的可能
for i in [580,725,1064]:
    BsmtCond_count=Bsmt_data.loc[(Bsmt_data['BsmtFinType1']==Bsmt_data.loc[i,'BsmtFinType1'])&(Bsmt_data['BsmtExposure']==Bsmt_data.loc[i,'BsmtExposure'])&
                                 (Bsmt_data['BsmtQual']==Bsmt_data.loc[i,'BsmtQual'])&(Bsmt_data['BsmtFinType2']==Bsmt_data.loc[i,'BsmtFinType2'])]['BsmtCond'].value_counts()
    Bsmt_data.loc[i,'BsmtCond']=BsmtCond_count.index[BsmtCond_count.argmax()]
print(f'BsmtCond_Nan最有可能的值是{BsmtCond_count.index[BsmtCond_count.argmax()]}')


# In[ ]:


for i in [27,888]:
    BsmtExposure_count=Bsmt_data.loc[(Bsmt_data['BsmtFinType1']==Bsmt_data.loc[i,'BsmtFinType1'])&(Bsmt_data['BsmtCond']==Bsmt_data.loc[i,'BsmtCond'])&
                                 (Bsmt_data['BsmtQual']==Bsmt_data.loc[i,'BsmtQual'])&(Bsmt_data['BsmtFinType2']==Bsmt_data.loc[i,'BsmtFinType2'])]['BsmtExposure'].value_counts()
    Bsmt_data.loc[i,'BsmtExposure']=BsmtExposure_count.index[BsmtExposure_count.argmax()]
print(f'BsmtExposure_Nan最有可能的值是{BsmtExposure_count.index[BsmtExposure_count.argmax()]}')


# In[ ]:


for i in [757,758]:
    BsmtQual_count=Bsmt_data.loc[(Bsmt_data['BsmtFinType1']==Bsmt_data.loc[i,'BsmtFinType1'])&(Bsmt_data['BsmtCond']==Bsmt_data.loc[i,'BsmtCond'])&
                                 (Bsmt_data['BsmtExposure']==Bsmt_data.loc[i,'BsmtExposure'])&(Bsmt_data['BsmtFinType2']==Bsmt_data.loc[i,'BsmtFinType2'])]['BsmtQual'].value_counts()
    Bsmt_data.loc[i,'BsmtQual']=BsmtCond_count.index[BsmtQual_count.argmax()]
print(f'BsmtQual_Nan最有可能的值是{BsmtQual_count.index[BsmtQual_count.argmax()]}')


# In[ ]:


Bsmt_data=Bsmt_data.fillna(value='No_data')
test[['BsmtFinType2','BsmtExposure','BsmtQual','BsmtCond','BsmtFinType1']]=Bsmt_data
test=test.fillna(value='No_data')
print(test.isnull().sum())


# ok! now test is no missing value.

# # *Building Linear Model*

# In[ ]:


#Build Function
#Standard
def ss(data):
    ss=StandardScaler()
    data_ss=pd.DataFrame(ss.fit_transform(data))
    data_ss.columns=data.columns
    return data_ss
#Linear Regression
def moele_lin(x_data,y_data,test_size):
    X_train ,X_test ,y_train ,y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=0) 
    print(X_train.shape,X_test.shape)
    print(y_train.shape,y_test.shape)
    
    model_linear = LinearRegression()
    model_linear.fit(X_train, y_train)

    y_pred_linear = model_linear.predict(X_test)
    
    print(f'MAE：{mean_absolute_error(y_test, y_pred_linear):.3f}\n'
          f'MSE：{mean_squared_error(y_test, y_pred_linear):.3f}\n'
          f'R-square：{r2_score(y_test, y_pred_linear):.3f}')
    
    plt.figure(figsize=(16,9)) #尺吋
    plt.plot(range(len(y_test)), y_test, 'X-', label="Real") #plot折線圖
    plt.plot(range(len(y_pred_linear)), y_pred_linear, 'o-', label="Pred")
    plt.legend() #加圖例
    plt.ylabel("total_bill")
    plt.show()
#Ridge
def moele_Rid(x_data,y_data,test_size):
    X_train ,X_test ,y_train ,y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=0) 
    print(X_train.shape,X_test.shape)
    print(y_train.shape,y_test.shape)
    
    moele_Ridge = Ridge(alpha=0.5)
    moele_Ridge.fit(X_train, y_train)

    y_pred_Ridge = moele_Ridge.predict(X_test)
    
    print(f'MAE：{mean_absolute_error(y_test, y_pred_Ridge):.3f}\n'
          f'MSE：{mean_squared_error(y_test, y_pred_Ridge):.3f}\n'
          f'R-square：{r2_score(y_test, y_pred_Ridge):.3f}')
    
    plt.figure(figsize=(16,9)) #尺吋
    plt.plot(range(len(y_test)), y_test, 'X-', label="Real") #plot折線圖
    plt.plot(range(len(y_pred_Ridge)), y_pred_Ridge, 'o-', label="Pred")
    plt.legend() #加圖例
    plt.ylabel("total_bill")
    plt.show()
#Lasso
def modle_Lasso(x_data,y_data,test_size):
    X_train ,X_test ,y_train ,y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=0) 
    print(X_train.shape,X_test.shape)
    print(y_train.shape,y_test.shape)
    
    moele_Lasso = Lasso(alpha=0.5)
    moele_Lasso.fit(X_train, y_train)

    y_pred_Lasso = moele_Lasso.predict(X_test)
    
    print(f'MAE：{mean_absolute_error(y_test, y_pred_Lasso):.3f}\n'
          f'MSE：{mean_squared_error(y_test, y_pred_Lasso):.3f}\n'
          f'R-square：{r2_score(y_test, y_pred_Lasso):.3f}')
    
    plt.figure(figsize=(16,9)) #尺吋
    plt.plot(range(len(y_test)), y_test, 'X-', label="Real") #plot折線圖
    plt.plot(range(len(y_pred_Lasso)), y_pred_Lasso, 'o-', label="Pred")
    plt.legend() #加圖例
    plt.ylabel("total_bill")
    plt.show()
#SVR
def model_svr(x_data,y_data,test_size):
    X_train ,X_test ,y_train ,y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=0) 
    print(X_train.shape,X_test.shape)
    print(y_train.shape,y_test.shape)
    model_svr = SVR(C=2,kernel='linear')
    model_svr.fit(X_train, y_train)

    y_pred_svr = model_svr.predict(X_test)
    
    print(f'MAE：{mean_absolute_error(y_test, y_pred_svr):.3f}\n'
          f'MSE：{mean_squared_error(y_test, y_pred_svr):.3f}\n'
          f'R-square：{r2_score(y_test, y_pred_svr):.3f}')
    
    plt.figure(figsize=(16,9)) #尺吋
    plt.plot(range(len(y_test)), y_test, 'X-', label="Real") #plot折線圖
    plt.plot(range(len(y_pred_svr)), y_pred_svr, 'o-', label="Pred")
    plt.legend() #加圖例
    plt.ylabel("total_bill")
    plt.show()


# # **NO** StandardScaler & Select features

# In[ ]:


#train_columns no match test_columns
#train_columns=1460
#test_columns=1459
#train need to drop one row
#456 is randem i select
train=train.drop(456,axis=0)
train = pd.get_dummies(train)
test = pd.get_dummies(test)


# In[ ]:


x=train.drop("SalePrice",axis=1)
y=train["SalePrice"]


# In[ ]:


#linear
moele_lin(x,y,0.3)


# In[ ]:


#Ridge
moele_Rid(x,y,0.3)


# In[ ]:


#Lasso
modle_Lasso(x,y,0.3)


# In[ ]:


#SVR
#model_svr(x,y,0.3)


# # StandardScaler & Select features

# In[ ]:


train_ss=ss(train)


# In[ ]:


#找出與目標變數SalePrice據有線性正相關強度前10個特徵
k=11
corr = train_ss.corr()
cols_largest=corr.nlargest(k,'SalePrice')['SalePrice'].index
sns.set(font_scale=1.25)
plt.figure(figsize=(16,10))
cm = np.corrcoef(train_ss[cols_largest].values.T)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols_largest.values, xticklabels=cols_largest.values)
plt.show()


# In[ ]:


#篩選特徵
cols_largest=pd.Series(cols_largest)
train_ss_cols_largest= pd.DataFrame(train_ss,columns=cols_largest).drop('SalePrice',axis=1)


# In[ ]:


#找出與目標變數SalePrice據有線性負相關強度前10個特徵
k=10
cols_smallest=corr.nsmallest(k, 'SalePrice')['SalePrice'].index
cols_smallest=cols_smallest.insert(0, 'SalePrice')
sns.set(font_scale=1.25)
plt.figure(figsize=(16,10))
cm = np.corrcoef(train_ss[cols_smallest].values.T)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols_smallest.values, xticklabels=cols_smallest.values)
plt.show()


# In[ ]:


#篩選特徵
cols_smallest=pd.Series(cols_smallest)
train_ss_cols_smallest= pd.DataFrame(train_ss,columns=cols_smallest).drop('SalePrice',axis=1)


# In[ ]:


train_ss_selected=pd.concat([train_ss_cols_largest,train_ss_cols_smallest],axis=1)
x=train_ss_selected
y=train['SalePrice']


# In[ ]:


moele_lin(x,y,0.3)
moele_Rid(x,y,0.3)
modle_Lasso(x,y,0.3)


# # Best_score_model

# In[ ]:


#moele_Ridge = Ridge(alpha=0.5)
#moele_Ridge.fit(x, y)

#y_pred_Ridge = moele_Ridge.predict(test)
    
#print(f'MAE：{mean_absolute_error(test, y_pred_Ridge):.3f}\n'
      #f'MSE：{mean_squared_error(test, y_pred_Ridge):.3f}\n'
      #f'R-square：{r2_score(test, y_pred_Ridge):.3f}')


# In[ ]:


#y_pred_Ridge.shape


# In[ ]:


#my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice' : y_pred_Ridge})
#my_submission.to_csv('submission.csv', index=False)

