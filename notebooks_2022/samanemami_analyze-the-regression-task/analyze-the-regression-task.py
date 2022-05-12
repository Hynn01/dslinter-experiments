#!/usr/bin/env python
# coding: utf-8

# # About this NoteBook
# 
# In this notebook, I review a multi-class classification problem.
# 
# In the following, you can find the explanation of the trained model, dataset, metric I used, and the focus of the study.
# 
# Also, as always, you can have free access to complete documentation of this NoteBook on my [Medium](https://samanemami.medium.com/) profile.
# 
# This Notebook only has the last version, and I do not update it.
# 
# ## The focus of this study 
# 
# The focus of this notebook is on applying a different statistical method to analyze the model's performance. 
# 
# <hr>
# 
# #### GitHub Package
# 
# To have access to the my package on GitHub, please refer to [here](https://github.com/samanemami/)
# 
# <hr>
# 
# 

# ### Author: [Seyedsaman Emami](https://github.com/samanemami)
# 
# If you want to have this method or use the outputs of the notebook, you can fork the Notebook as following (copy and Edit Kernel).
# 
# <img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1101107%2F8187a9b84c9dde4921900f794c6c6ff9%2FScreenshot%202020-06-28%20at%201.51.53%20AM.png?generation=1593289404499991&alt=media" alt="Copyandedit" width="300" height="300" class="center">
# 
# <hr>
# 
# ##### You can find some of my developments [here](https://github.com/samanemami?tab=repositories).
# 
# <hr>

# <a id='top'></a>
# # Contents
# 
# * [Importing libraries](#lib)
# * [Dataset](#dt)
#     * [Describe](#des)
#     * [Normalization](#Normalization)
# * [model](#model)

# <a id='lib'></a>
# # Importing libraries

# In[ ]:


import os
import numpy as np
import pandas as pd
import scipy.stats as st
import sklearn.datasets as dts
import matplotlib.pyplot as plt
from itertools import permutations
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import StandardScaler


# In[ ]:


import warnings
random_state = 123
np.random.seed(random_state)

warnings.simplefilter('ignore')

np.set_printoptions(precision=4, suppress=True)


# <a id='dt'></a>
# # Importing dataset

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        path = os.path.join(dirname, filename)
df = pd.read_csv(path)
df.head()


# <hr>
# 
# #### [Scroll Back To Top](#top)
# 
# <hr>

# <a id='des'></a>
# ## Describe the dataset

# In[ ]:


df.describe().T


# In[ ]:


X = (df.drop(columns=df[['Y1', 'Y2']], axis=0)).values
y = (df.iloc[:, -2:]).values

print('\n', 'X shape:',
      X.shape, '\n',
      'y shape:', y.shape)


# <a id='Normalization'></a>
# ## Normalization

# In[ ]:


scl = StandardScaler()
X = scl.fit_transform(X)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y)


# <hr>
# 
# #### [Scroll Back To Top](#top)
# 
# <hr>

# <a id='model'></a>
# # Model definition
# 

# In[ ]:


pred_mart = np.zeros_like(y_test)
mart = GradientBoostingRegressor(max_depth=5,
                                 subsample=0.75,
                                 max_features="sqrt",
                                 learning_rate=0.1,
                                 random_state=1,
                                 criterion="mse",
                                 n_estimators=100)
for i in range(y.shape[1]):
    mart.fit(x_train, y_train[:, i])
    pred_mart[:, i] = mart.predict(x_test)


# ## Kernel density estimation
# Knowing the probability density function of PDF, we need a tool to estimate it for a random continuous variable. 
# 
# Kernel density estimation (KDE) is a non-parametric for this matter.

# Here, we assume that we want to evaluate the result of the regression model, which we already have the predicted values. Therefore, we need to use the gaussian KDE from `scipy.stats.`

# In[ ]:


def three_d(x, y):
    deltaX = (max(x) - min(x))/10
    deltaY = (max(y) - min(y))/10

    xmin = min(x) - deltaX
    xmax = max(x) + deltaX

    ymin = min(y) - deltaY
    ymax = max(y) + deltaY

    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1,
                           cmap='coolwarm', edgecolor='none')
    ax.set_xlabel('predicted values')
    ax.set_ylabel('real values')
    ax.set_zlabel('PDF')
    ax.set_title('Surface plot of Gaussian 2D KDE')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(60, 35)
    plt.show()
    plt.close('all')

    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    w = ax.plot_wireframe(xx, yy, f)
    ax.set_xlabel('predicted value')
    ax.set_ylabel('real values')
    ax.set_zlabel('PDF')
    plt.tight_layout()
    ax.set_title('Wireframe plot of Gaussian 2D KDE')


# In[ ]:


if __name__=='__main__':
    x = pred_mart[:, 0]
    y = y_test[:, 0]
    three_d(x, y)

