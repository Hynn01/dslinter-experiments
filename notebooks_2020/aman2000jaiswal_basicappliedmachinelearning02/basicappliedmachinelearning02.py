#!/usr/bin/env python
# coding: utf-8

# # Visit BasicAppliedMachineLearning01
# https://www.kaggle.com/aman2000jaiswal/basicappliedmachinelearning01
# 

# # BasicAppliedMachineLearning02

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Synthetic dataset
# ****Some of the synthetic dataset provided by sklearn.dataset bunch type ****
# It will be helpful for beginner to generate their own dataset quickly and practice
# before going to real world dataset.

# In[ ]:


from sklearn.datasets import make_regression
plt.figure(figsize=(10,8))
plt.title("regression dataset example")
X_R1,y_R1=make_regression(n_samples=100,n_features=1,n_informative=1,bias=150.0,noise=30,random_state=0)
'''
The number of informative features, i.e., the number of features used to build the linear model used to generate the output.

'''
plt.grid()
plt.scatter(X_R1,y_R1,marker='o',s=50)
plt.show()


# In[ ]:


from matplotlib.colors import ListedColormap
cmap_bold=ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])


# In[ ]:


from sklearn.datasets import make_classification
plt.figure(figsize=(8,6))
plt.title('simple classification example')
X_C1,y_C1=make_classification(n_samples=100,n_features=2,n_informative=2,n_redundant=0,n_classes=2,n_clusters_per_class=1,flip_y=0.1,class_sep=0.5,random_state=0)
plt.grid()
plt.scatter(X_C1[:,0],X_C1[:,1],c=y_C1,marker='o',s=50,cmap=cmap_bold)
plt.show()


# In[ ]:


from sklearn.datasets import make_blobs
X_D2,y_D2=make_blobs(n_samples=100,n_features=2,centers=8,cluster_std=1.3,random_state=4)
y_D2=y_D2%2
plt.figure(figsize=(12,8))
plt.title('complex classification example ')
plt.scatter(X_D2[:,0],X_D2[:,1],c=y_D2,s=50,marker='o',cmap=cmap_bold)
plt.grid()
plt.show()


# In[ ]:


X,y=make_regression(n_samples=60,n_features=1,n_targets=1,n_informative=1,bias=10,noise=20,random_state=0)
from sklearn.neighbors import KNeighborsRegressor
knn_reg=KNeighborsRegressor(n_neighbors=5)# as n_neighbors increase model_complexity will reduce and reduce overfitting 
knn_reg.fit(X[:40],y[:40])
plt.figure(figsize=(13,8))
plt.scatter(X[:40],y[:40],marker='o',s=50,label='training_set')

plt.plot(np.linspace(min(X[:40]),max(X[:40]),num=500),knn_reg.predict(np.linspace(min(X[:40]),max(X[:40]),num=500)),'r-',label='knn_model')
plt.plot(X[40:],knn_reg.predict(X[40:]),'go',label='Test_prediction')
plt.scatter(X[40:],y[40:],marker='*',s=50,c='y',label='Test_set')
plt.title('Knn regressor, k=5')
plt.legend(loc='best')
plt.grid()
plt.show()
print('train_socre ',knn_reg.score(X[:40],y[:40]))
print('test_score ',knn_reg.score(X[40:],y[40:]))


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
knn_reg=KNeighborsRegressor(n_neighbors=10)   
knn_reg.fit(X[:40],y[:40])
plt.figure(figsize=(10,6))
plt.scatter(X[:40],y[:40],marker='o',s=50,label='training_set')

plt.plot(np.linspace(min(X[:40]),max(X[:40]),num=500),
         knn_reg.predict(np.linspace(min(X[:40]),max(X[:40]),num=500)),'r-',label='knn_model')
plt.plot(X[40:],knn_reg.predict(X[40:]),'go',label='Test_prediction')
plt.scatter(X[40:],y[40:],marker='*',s=50,c='y',label='Test_set')
plt.title('Knn regressor, k=10')
plt.legend(loc='best')
plt.grid()
plt.show()
print('train_socre ',knn_reg.score(X[:40],y[:40]))
print('test_score ',knn_reg.score(X[40:],y[40:]))


# In[ ]:


from sklearn.linear_model import LinearRegression
linear_model=LinearRegression()
linear_model.fit(X[:40],y[:40])
plt.figure(figsize=(10,6))
plt.scatter(X[:40],y[:40],s=50,marker='o',label='training_set')
plt.plot(X[:40],(linear_model.coef_)*X[:40]+linear_model.intercept_,'r-',label='linear_model')
plt.plot(X[40:],(linear_model.coef_)*X[40:]+linear_model.intercept_,'go',label='test_prediction')
plt.scatter(X[40:],y[40:],s=50,marker='*',c='y',label='test_set')
plt.title('linear_regressor')
plt.legend(loc='best')
plt.grid()
plt.show()
print('intercept  ',linear_model.intercept_)
print('coef  ',linear_model.coef_)
print('train_score ',linear_model.score(X[:40],y[:40]))
print('test_score',linear_model.score(X[40:],y[40:]))


# # RIDGE REGRESSION
#  Ridge Regression use l2 penality, It penalize the model to become over weight. If model having different weights having same accuracy or prediction then it make tends to choose less weight model. 
#  Penality term is given as :
#                                 n
#             L2_Penality = alpha*∑ w^2
#                                 j=1 
#                                 
#  as alpha increase model will tends to simpler model and generalization.
#  It reduces the overfitting of model.
#  
# 

# In[ ]:


from sklearn.linear_model import Ridge
linridge=Ridge(alpha=2)       
linridge.fit(X[:40],y[:40])
print('train_score ',linridge.score(X[:40],y[:40]))
print('test_score ',linridge.score(X[40:],y[40:]))


# # LASSO REGRESSION
#     Lasso Regression uses L1 Penality.
# 
#                                 n
#             L1_Penality = alpha*∑ w
#                                 j=1 
#    as alpha increase model will tends to simpler model and generalization. It reduces the overfitting of model.
#    The difference between ridge and lasso is that ridge Penality tends weight to zero. But lasso make weight zero of the    less relvalnt features of dataset  

# 

# In[ ]:


from sklearn.linear_model import Lasso
linlasso=Lasso(alpha=1,max_iter=2000)
linlasso.fit(X[:40],y[:40])
print('train_score ',linlasso.score(X[:40],y[:40]))
print('test_score ',linlasso.score(X[40:],y[40:]))


# # MINMAXSCALER
# minmaxscaler convert features to range 0 to 1.some model which work better with features scaling.
# 
# In general, algorithms that exploit distances or similarities (e.g. in form of scalar product) between data samples, such as k-NN and SVM, are sensitive to feature transformations.
# 
# Graphical-model based classifiers, such as Fisher LDA or Naive Bayes, as well as Decision trees and Tree-based ensemble methods (RF, XGB) are invariant to feature scaling, but still it might be a good idea to rescale/standartize your data.
# 
# Minmaxscaler prone to outliers.So the dataset must remove outliers for effective use of minmaxscaler.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaled_X_train=scaler.fit_transform(X[:40])
scaled_X_test=scaler.transform(X[40:])
linear_model2=LinearRegression()
linear_model2.fit(scaled_X_train,y[:40])
print('linear model train_score ',linear_model2.score(scaled_X_train,y[:40]))
print('linear model test_score',linear_model2.score(scaled_X_test,y[40:]))


# # **Polynomial features**
# Some of the dataset which are not linear. For those complex dataset we have to add some of the quadratic or polynomial features
# value by raising the degree of the features.
# Such that features X to X^2 or X^3.
# For this purpose we transform the dataset to polynomial features which gives all the possible specified degree
# features to the features set.

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
poly_X=poly.fit_transform(X)
linear_model3=LinearRegression()
linear_model3.fit(poly_X[:40],y[:40])
print('linear model train_score ',linear_model3.score(poly_X[:40],y[:40]))
print('linear model test_score',linear_model3.score(poly_X[40:],y[40:]))

