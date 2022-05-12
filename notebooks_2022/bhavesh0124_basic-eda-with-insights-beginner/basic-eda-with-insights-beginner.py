#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import statsmodels.api as sm
import calendar

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv')
submission  = pd.read_csv('../input/tabular-playground-series-may-2022/sample_submission.csv')


# ## Check the uniquness of ID

# In[ ]:


print("The percentage of unique id records are ", train['id'].unique().shape[0] / train.shape[0] *100)


# ## Overall feature distribution

# In[ ]:


combined=pd.concat([train,test],axis=0)


# In[ ]:


features = list(combined.drop(['f_27'],axis=1).columns[1:-1])
fig = plt.figure(figsize = (20,15))
ax = fig.gca()
train[features].hist(ax=ax, color='r')
plt.show()


# #### Some comments about the features:
# 1. f_29 looks liks a categorical with two values 0 and 1
# 2. f_30 looks like cateogrical columns with 0,1,2 value
# 3. f_25, f_20, f_26, f_28, f_29, f_30, f_0 : f_06 holds a bell shape curve which is a good sign and means that they have been normalized
# 4. The rets of features looks like a bit left skewed
# 5. The ranges of features are almost same, where they range somewhere around 0, except for f_28, so we can perform some scaling like standard scaler or min max scaler

# ## Correlation

# In[ ]:


mask = np.zeros_like(combined[features].corr(), dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.set(rc={'figure.figsize':(30,20)})
sns.heatmap(combined[features].corr(),
            annot=True,
            mask = mask,
            cmap = 'RdBu_r',
            linewidths=0.1, 
            linecolor='white',
            vmax = .9,
            square=True)
plt.show()


# ## Dealing with feature f_27

# In[ ]:


print("The percentage of unique values in f_27 is ", len(combined['f_27'].unique()) / len(combined))


# In[ ]:


from collections import Counter
count_dict = Counter(combined['f_27'])
print("The top repeating element's frequency in the column is")
print(sorted(count_dict.values(), reverse = True)[:10])


# ## Final Insights:
# 1. Doesnot look like there a correlation in the features, so we can ignore this fact and not drop any features
# 2. we can do a feature scaling before feeding to the neural network model which ill cover in another notebook
# 3. we can hot encode, f_29, f_30 since they look categorical
# 4. Drop the column f_27, since it contains a lot of unique values, and the frequency of the max repeating element is 15
