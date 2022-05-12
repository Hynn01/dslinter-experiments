#!/usr/bin/env python
# coding: utf-8

# ## Torch 数据集  怎么制作 （九天，菜菜的课件）
# ### 二维表格变成四位tensor  
# - 可以通过升维的方式  (from sklearn.preprocessing import PolynomialFeatures as PF  ) 
# - 也不是所有的数据都可以升维，需要用代码来测试 
# 
# #### 加利福利亚FCH 数据集在升维后有过拟合现象

# In[ ]:


from sklearn.datasets import fetch_california_housing as FCH 
from sklearn.preprocessing import PolynomialFeatures as PF  
from sklearn.linear_model import  LinearRegression as LG  
from sklearn.model_selection import train_test_split as TTS  
from sklearn.metrics import mean_squared_error as MSE  


# In[ ]:


data = FCH()  


# In[ ]:


X = data.data
y = data.target  
Xtrain,Xtest,Ytrain,Ytest = TTS(X,y,test_size=0.3,random_state=115)
reg = LG().fit(Xtrain,Ytrain)
print(MSE(reg.predict(Xtrain),Ytrain))
print(MSE(reg.predict(Xtest),Ytest))


# - 过拟合
#     - 看到升维后出现了过拟合

# In[ ]:


poly = PF(degree=4).fit(Xtrain)
Xtrain_ = poly.transform(Xtrain)
Xtest_ = poly.transform(Xtest)

print(Xtrain_.shape)
reg = LG().fit(Xtrain_,Ytrain)

print(MSE(reg.predict(Xtrain_),Ytrain))
print(MSE(reg.predict(Xtest_),Ytest))


# #### 

# ### 封面数据集升维   

# In[ ]:


from sklearn.datasets import fetch_covtype as FC 
from sklearn.linear_model import  LogisticRegression as LR 

data = FC()  

print(data.data.shape)
print(data.target)


# In[ ]:


X=data.data[:2000]
print(X.shape)

y = data.target[:2000]
Xtrain,Xtest,Ytrain,Ytest = TTS(X,y,test_size=0.3,random_state=115)
# 参数 solver='newton-cg' 表示选用的牛顿法 做优化迭代
clf = LR(random_state=115,max_iter=1000,solver='newton-cg').fit(Xtrain,Ytrain)


# In[ ]:


print(clf.score(Xtrain,Ytrain))
print(clf.score(Xtest,Ytest))

poly = PF(degree=2,interaction_only=True).fit(Xtrain) # 不包含各特征的平方项


Xtrain_ = poly.transform(Xtrain)
Xtest_ = poly.transform(Xtest)


# In[ ]:


Xtrain_.shape 


# In[ ]:


clf_ = LR(random_state=115,max_iter=1000,solver='newton-cg').fit(Xtrain_,Ytrain)

print(clf_.score(Xtrain_,Ytrain))
print(clf_.score(Xtest_,Ytest))


# - 看到升维后的 准确度还不错 。 
# #### 那么接下来对这个数据 转换成四维Tensor 操作 
# - 假如网络模型接收 32*32 的形状  
# - 封面的数据集 是7 分类  

# In[ ]:


clf_.coef_.shape  


# In[ ]:


import numpy as np 
import pandas as pd 

pd_weight = pd.DataFrame(abs(clf_.coef_).mean(axis=0),columns=['weight'])
idx = pd_weight.sort_values(by='weight',ascending=False).iloc[:1024].index
idx  


# - 升维取所有行 

# In[ ]:


X_ = poly.transform(X)
print(X_[:,idx].shape)
X_torch = X_[:,idx].reshape(2000,1,32,32)
X_torch.shape 


# In[ ]:


import torch 
from torch.utils.data import TensorDataset   

data = TensorDataset(torch.tensor(X_torch),torch.tensor(y))
data  


# In[ ]:


for x, y in data:
    print(x.shape)
    print(y)
    break  


# - 这几就可以向tensor 一样索引任何数据 

# In[ ]:


data[6]

