#!/usr/bin/env python
# coding: utf-8

# **Time Complexity of Machine Learning Algorithms**
# 
# Table of Contents:
# 1. Hard computing vs Soft computing
# 1. A Theoretical point of view
# 1. Justifications
# 1. A Practical point of view
# 1. Algorithm Complexity 
# 

# # 1. Hard computing vs Soft computing 
# There are two paradigms of computing - hard computing and soft computing.
# 
# Hard computing deals with problems that have exact solutions, and in which approximate / uncertain solutions are not acceptable. This is the conventional computing, and most algorithms courses deal with hard computing.
# 
# Soft computing, on the other hand, looks at techniques to approximately solve problems that are not solvable in finite time. Most machine learning algorithms fall in this category. The quality of the solution improves as you spend more time in solving the problem. So complexity in terms of big-oh ,we can make appropriate  but 

# # 2. A Theoretical point of view
# It is harder than one would think to evaluate the complexity of a machine learning algorithm, especially as it may be implementation dependent, properties of the data may lead to other algorithms or the training time often depends on some parameters passed to the algorithm.
# 
# Lets start looking at worst case time complexity when the data is dense,
# * n -> the number of training sample
# * p -> the number of features
# * n*tress* -> the number of trees (for methods based on various trees)
# * n*sv* -> the number of support vectors
# * n*li* -> the number of neurons at layer i in a neural network
# 
# 
# 
# we have the following approximations,

# In[ ]:


from IPython.display import Image
Image("../input/ml-cplexity/ml.JPG")


# # 3. Justifications
# ### Decision Tree based models
# 
# 
# 
# Obviously, ensemble methods multiply the complexity of the original model by the number of “voters” in the model, and replace the training size by the size of each bag.
# 
# When training a decision tree, a split has to be found until a maximum depth 
# d
#  has been reached.
# 
# The strategy for finding this split is to look for each variable (there are 
# p
#  of them) to the different thresholds (there are up to 
# n
#  of them) and the information gain that is achieved (evaluation in 
# O
# (
# n
# )
# )
# 
# In the Breiman implementation, and for classification, it is recommanded to use 
# √
# p
#  predictors for each (weak) classifier.
# 
# 
# ### Linear regressions
# 
# The problem of finding the vector of weights 
# β
#  in a linear regression boils down to evaluating the following equation: 
# β=
# (
# X
# ′
# X
# )
# −
# 1
# X
# ′
# Y
# .
# 
# The most computationnaly intensive part is to evaluate the product 
# X
# ′
# X
# , which is done in 
# p
# 2
# n
#  operations, and then inverting it, which is done in 
# p
# 3
#  operations.
# 
# Though most implementations prefer to use a gradient descent to solve the system of equations 
# (
# X
# ′
# X
# )
# β=
# X
# ′
# Y
# , the complexity remains the same.
# 
# 
# Support Vector Machine
# 
# For the training part, the classical algorithms require to evaluate the kernel matrix 
# K
# , the matrix whose general term is 
# K
# (
# x
# i
# ,
# x
# j
# )
#  where 
# K
#  is the specified kernel.
# 
# It is assumed that K can be evaluated with a 
# O
# (
# p
# )
#  complexity, as it is true for common kernels (Gaussian, polynomials, sigmoid…). This assumption may be wrong for other kernels.
# 
# Then, solving the constrained quadratic programm is “morally equivalent to” inverting a square matrix of size 
# n
# , whose complexity is assumed to be 
# O
# (
# n
# 3
# )
# 
# ### k-Nearest Neighbours
# 
# In its simplest form, given a new data point 
# x
# , the kNN algorithm looks for the k closest points to 
# x
#  in the training data and returns the most common label (or the averaged values of targets for a regression problem).
# 
# To achieve this, it is necessary to compare the distance between 
# x
#  and every point in the data set. This amounts to 
# n
#  operations. For the common distances (Euclide, Manhattan…) this operation is performed in a 
# O
# (
# p
# )
#  operations. Not that kernel k Nearest Neighbours have the same complexity (provided the kernels enjoy the same property).
# 
# However, many efforts pre-train the kNN, indexing the points with quadtrees, which enable to lower dramatically the number of comparisons to the points in the training set.
# 
# Likewise, in the case of a sparse data set, with an inverted indexation of the rows, it is not necessary to compute all the pairwise distances.

# # 4. The practical point of view
# The assumptions will be that the complexities take the form of 
# O
# (
# n
# α
# p
# β
# )
#  and 
# α
#  and 
# β
#  will be estimated using randomly generated samples with 
# n
#  and 
# p
#  varying. Then, using a log-log regression, the complexities are estimated.
# 
# Though this assumption is wrong, it should help to have a better idea of how the algorithms work and it will reveal some implementation details / difference between the default settings of the same algorithm that one may overlook.

# Another interesting point to note are the complexities in 
# p
#  for the random forest and extra trees, the component in 
# p
#  varies according to the fact that we are performing a regression or a classification problem. A short look at the documentation explains it, they have different behaviors for each problem!
#  
#  **For the regression:**
#  max_features : int, float, string or None, optional (default=”auto”)
# 
# The number of features to consider when looking for the best split:
# 
# * If int, then consider max_features features at each split.
# * If float, then max_features is a percentage and int(max_features * n_features) features are considered at each split.
# * If “auto”, then max_features=n_features.
# * If “sqrt”, then max_features=sqrt(n_features).
# * If “log2”, then max_features=log2(n_features).
# * If None, then max_features=n_features.
# 
# Whereas the classification default behavior is
# * If “auto”, then max_features=sqrt(n_features).
# 
# To learn more on this,
# * [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
# * [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

# 

# In[ ]:



import time
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# In[ ]:


class ComplexityEvaluator:
    def __init__(self , nrow_samples , ncol_samples):
        self._nrow_samples = nrow_samples
        self._ncol_samples = ncol_samples
    
    #random data
    def _time_samples(self , model , random_data_generator):
        row_list = []
        # iterate with rows and columns
        for nrow in self._nrow_samples:
            for ncol in self._ncol_samples:
                train , label = random_data_generator(nrow , ncol)
                #initiate timer
                start_time = time.time()
                model.fit(train , label)
                elapsed_time = time.time() - start_time
                result = {"N" : nrow , "P" : ncol , "Time" : elapsed_time}
                row_list.append(result)
                
        return row_list , len(row_list)
    
    #house pricing data
    def _time_houseprice(self , model):
        row_list = []
        #initiate timer
        train = self._nrow_samples
        label = self._ncol_samples
        start_time = time.time()
        model.fit(train , label)
        elapsed_time = time.time() - start_time
        #print("time : " , elapsed_time)
        result = {"N" :len(self._nrow_samples) , "P" : len(self._ncol_samples), "Time" : elapsed_time}
        row_list.append(result)
                
        return row_list , len(row_list)
    
    def run(self , model , random_data_generator , ds='random'):
        import random
        if ds == 'random':
            row_list , length = self._time_samples(model, random_data_generator)
        else:
            row_list , length = self._time_houseprice(model)
            
        cols = list(range(0 , length))
        data = pd.DataFrame(row_list , index =cols)
        print(data)
        data = data.applymap(math.log)
        #print("apply math : ", data)
        linear_model = LinearRegression(fit_intercept=True)
        linear_model.fit(data[["N" , "P"]] , data[["Time"]])
        #print("coefficients : " , linear_model.coef_)
        return linear_model.coef_
        


# In[ ]:


class TestModel:
    def __init__(self):
        pass
    
    def fit(self , x, y):
        time.sleep(x.shape[0] /1000)
        


# In[ ]:


def random_data_generator(n , p):
    return np.random.rand(n , p) , np.random.rand(n , 1)


# After a small unit test, everything seems consistent.

# In[ ]:


if __name__ == "__main__":
    model = TestModel()
    nrow_samples = [200, 500, 1000, 2000, 3000]
    ncol_samples = [1,5,10]
    complexity_evaluator = ComplexityEvaluator(nrow_samples , ncol_samples)
    res = complexity_evaluator.run(model , random_data_generator)


# So let’s enjoy the number of algorithms offered by sklearn.I have used House price data to run on different ML models

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LogisticRegression


# In[ ]:


regression_models = [RandomForestRegressor(),
                     ExtraTreesRegressor(),
                     AdaBoostRegressor(),
                     LinearRegression(),
                     SVR()]

classification_models = [RandomForestClassifier(),
                         ExtraTreesClassifier(),
                         AdaBoostClassifier(),
                         SVC(),
                         LogisticRegression(),
                         LogisticRegression(solver='sag')]


# In[ ]:


names = ["RandomForestRegressor",
         "ExtraTreesRegressor",
         "AdaBoostRegressor",
         "LinearRegression",
         "SVR",
         "RandomForestClassifier",
         "ExtraTreesClassifier",
         "AdaBoostClassifier",
         "SVC",
         "LogisticRegression(solver=liblinear)",
         "LogisticRegression(solver=sag)"]


# In[ ]:


#using sample data to run on different models
sample_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
sample_data = sample_data.loc[:, sample_data.dtypes !=np.object]
sample_data = sample_data.fillna(0)
nrows = sample_data.iloc[:,:-1].values.tolist()
ncols = sample_data['SalePrice'].values.tolist()
complexity_evaluator = ComplexityEvaluator(nrows,ncols)


# In[ ]:


i = 0
for model in regression_models:
    res = complexity_evaluator.run(model, random_data_generator , 'houseprice')[0]
    print(names[i] + ' | ' + str(round(res[0], 2)) +
          ' | ' + str(round(res[1], 2)))
    i = i + 1


# # 5. Algorithm Complexity
# 
# Machine Learning is primarily about optimization of an objective function. Often the function is so represented that the target is to reach the global minima. Solving it involves heuristics, and thereby multiple iterations. In gradient descent for instance, you need multiple iterations to reach the minima. So given an algorithm, you can at best estimate the running 'time' for a single iteration. 
# 
# We are talking about finding Minima of cost functions whose complexity depend on the ‘value’ of the data and not just the ‘size’ of the data. The cost function is a function of the dataset. This is a key difference between algorithms used for ML and others.
# 
# Note that this again cannot be a parameter for comparison since for different algorithms, the objective function would reach a minima in different number of iterations for different data sets.
# 

# **References:**
# * [https://mitpress.mit.edu/books/introduction-computational-learning-theory](http://https://mitpress.mit.edu/books/introduction-computational-learning-theory)
# * [https://mitpress.mit.edu/books/computational-complexity-machine-learning](http://https://mitpress.mit.edu/books/computational-complexity-machine-learning)
# * [http://www.scalaformachinelearning.com/2015/11/time-complexity-in-machine-learning.html](http://http://www.scalaformachinelearning.com/2015/11/time-complexity-in-machine-learning.html)
# * [https://scikit-learn.org/stable/modules/manifold.html](http://https://scikit-learn.org/stable/modules/manifold.html)
