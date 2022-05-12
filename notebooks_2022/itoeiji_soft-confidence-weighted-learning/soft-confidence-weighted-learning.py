#!/usr/bin/env python
# coding: utf-8

# <h1><span style="background-color: cyan">Develop online machine learning.</span></h1>
# <h3><span>In this notebook, develop a soft-confidence weighted learning, which is online machine learning. To easy test, use iris dataset.</span></h3>

# <h3><span style="background-color: cyan">Soft-Confidence Weighted learning, version Prop.1</span></h3>
# <h3><span>See: <a href="http://icml.cc/2012/papers/86.pdf">http://icml.cc/2012/papers/86.pdf</a></span></h3>

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import datasets
from scipy import stats


# In[ ]:


# Use iris dataset

iris = datasets.load_iris()
data = pd.DataFrame(data=np.c_[iris["data"], iris["target"]], columns= iris["feature_names"] + ["target"])
selected_species = [0, 1] # setosa, versicolor

df = data[data["target"].isin(selected_species)] # setosa, versicolor data
df = df.reindex(np.random.permutation(df.index)) # shuffle

data_x = df[iris["feature_names"]].values # sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)
data_t = [1 if i == selected_species[0] else -1 for i in df["target"]] # label = {1, -1}

train_x, test_x = data_x[:80], data_x[80:]
train_t, test_t = data_t[:80], data_t[80:]

print("Number of train data: ", len(train_x))
print("Number of test data: ", len(test_x))


# In[ ]:


# Class soft confidence weighted learning prop1
class SCW1():
    def __init__(self, in_size):
        self.mu = np.zeros(in_size)
        self.sigma = np.eye(in_size)
        self.eta = 0.95
        self.C = 1
        self.phi = stats.norm.ppf(self.eta)
        self.psi = 1+self.phi**2/2
        self.xi = 1+self.phi**2
    
    def train(self, x, t):
        m_t = t*self.mu.dot(x)
        v_t = x.dot(self.sigma).dot(x)
        alpha_t = min(self.C, max(0, (-m_t*self.psi+np.sqrt((m_t**2)*(self.phi**4)/4+v_t*(self.phi**2)*self.xi))/(v_t*self.xi)))
        u_t = ((-alpha_t*v_t*self.phi+np.sqrt((alpha_t**2)*(v_t**2)*(self.phi**2)+4*v_t))**2)/4
        beta_t = (alpha_t*self.phi)/(np.sqrt(u_t)+v_t*alpha_t*self.phi)
        self.mu = self.mu+alpha_t*t*self.sigma.dot(x)
        self.sigma = self.sigma-beta_t*self.sigma.dot(x)[:,np.newaxis].dot(x.dot(self.sigma)[np.newaxis,:])
        
    def predict(self, x):
        if x.dot(self.mu) > 0:
            return 1
        else:
            return -1


# In[ ]:


# Calc. correct answer rate
def get_accuracy(model, dataset_x, dataset_t):
    result = []
    for x, t in zip(dataset_x, dataset_t):
        if model.predict(x) * t > 0: # 1 if correct answer, -1 if incorrect answer
            result.append(1)
        else:
            result.append(0)
        accuracy = sum(result)/len(result)
        return accuracy


# In[ ]:


# train, test

EPOCH_NUM = 1
scw1 = SCW1(in_size=len(iris["feature_names"]))
for epoch in range(EPOCH_NUM):
    for x, t in zip(train_x, train_t):
        scw1.train(x, t)
    accuracy1 = get_accuracy(scw1, train_x, train_t) # Accuracy rate of train 
    accuracy2 = get_accuracy(scw1, test_x, test_t) # Accuracy rate of test
    print("train/accuracy: {}, test/accuracy: {}".format(accuracy1, accuracy2)) # Log


# <h3><span style="background-color: cyan;">Soft-Confidence Weighted learning, version Prop.2</span></h3>
# <h3><span>See: <a href="http://icml.cc/2012/papers/86.pdf">http://icml.cc/2012/papers/86.pdf</a></span></h3>

# In[ ]:


# Class soft confidence weighted learning prop2
class SCW2():
    
    def __init__(self, in_size):
        self.mu = np.zeros(in_size)
        self.sigma = np.eye(in_size)
        self.eta = 0.95
        self.C = 1
        self.phi = stats.norm.ppf(self.eta)
        self.psi = 1+self.phi**2/2
        self.xi = 1+self.phi**2
    
    def train(self, x, t):
        m_t = t*self.mu.dot(x)
        v_t = x.dot(self.sigma).dot(x)
        n_t = v_t+1/2*self.C
        gamma_t = self.phi*np.sqrt((self.phi**2)*(m_t**2)*(v_t**2)+4*n_t*v_t*(n_t+v_t*(self.phi**2)))
        alpha_t = max(0, (-(2*m_t*n_t+(self.phi**2)*m_t*v_t)+gamma_t)/(2*(n_t**2+n_t*v_t*(self.phi**2))))
        u_t = ((-alpha_t*v_t*self.phi+np.sqrt((alpha_t**2)*(v_t**2)*(self.phi**2)+4*v_t))**2)/4
        beta_t = (alpha_t*self.phi)/(np.sqrt(u_t)+v_t*alpha_t*self.phi)
        self.mu = self.mu+alpha_t*t*self.sigma.dot(x)
        self.sigma = self.sigma-beta_t*self.sigma.dot(x)[:,np.newaxis].dot(x.dot(self.sigma)[np.newaxis,:])
        
    def predict(self, x):
        if x.dot(self.mu) > 0:
            return 1
        else:
            return -1


# In[ ]:


# train, test

EPOCH_NUM = 1
scw2 = SCW2(in_size=len(iris["feature_names"]))
for epoch in range(EPOCH_NUM):
    for x, t in zip(train_x, train_t):
        scw2.train(x, t)
    accuracy1 = get_accuracy(scw2, train_x, train_t) # Accuracy rate of train 
    accuracy2 = get_accuracy(scw2, test_x, test_t) # Accuracy rate of test
    print("train/accuracy: {}, test/accuracy: {}".format(accuracy1, accuracy2)) # Log


# In[ ]:




