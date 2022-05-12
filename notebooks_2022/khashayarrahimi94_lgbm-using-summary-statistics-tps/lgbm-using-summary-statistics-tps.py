#!/usr/bin/env python
# coding: utf-8

# In[ ]:


conda install git


# In[ ]:


pip install git+https://github.com/OpenHydrology/lmoments3.git


# In[ ]:


get_ipython().system('pip uninstall -y lightgbm')
get_ipython().system('apt-get install -y libboost-all-dev')
get_ipython().system('git clone --recursive https://github.com/Microsoft/LightGBM')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'cd LightGBM\nrm -r build\nmkdir build\ncd build\ncmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..\nmake -j$(nproc)')


# In[ ]:


get_ipython().system('cd LightGBM/python-package/;python setup.py install --precompile')


# In[ ]:


get_ipython().system('mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd')
get_ipython().system('rm -r LightGBM')


# # Introduction
# 
# Tabular Playground Series is a time series competition, but here despite of using lags and other conventional approach for time series problem, I tried to use **Summary Statistics** for creating a dataset based on the TPS dataset.
# 
# In fact, I summarize the information of each sensor for each subject in some statistical measure like the following:
# 
# 1. Gini coefficient
# 2. L_moment
# 3. mean ($L_1$)
# 4. Variance
# 5. Standard Deviation
# 6. Mean Absolute Deviation
# 7. Interquartile Range
# 8. Trimmed Standard Deviation
# 9. Median Absolute Deviation
# 10. Scale
# 11. Skewness
# 12. Kurtosis

# In[ ]:


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import lmoments3 as lm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv("/kaggle/input/tabular-playground-series-apr-2022/train.csv")
test = pd.read_csv("/kaggle/input/tabular-playground-series-apr-2022/test.csv")
labels = pd.read_csv("/kaggle/input/tabular-playground-series-apr-2022/train_labels.csv")
sample_submission = pd.read_csv("/kaggle/input/tabular-playground-series-apr-2022/sample_submission.csv")
All = pd.concat([train, test], sort=True).reset_index(drop=True)


# # Gini coefficient
# 
# Here is the formula for gini coefficient:
# 
# ${\displaystyle G={\frac {\displaystyle {\sum _{i=1}^{n}\sum _{j=1}^{n}\left|x_{i}-x_{j}\right|}}{\displaystyle {2\sum _{i=1}^{n}\sum _{j=1}^{n}x_{j}}}}={\frac {\displaystyle {\sum _{i=1}^{n}\sum _{j=1}^{n}\left|x_{i}-x_{j}\right|}}{\displaystyle {2n\sum _{j=1}^{n}x_{j}}}}={\frac {\displaystyle {\sum _{i=1}^{n}\sum _{j=1}^{n}\left|x_{i}-x_{j}\right|}}{\displaystyle {2n^{2}{\bar {x}}}}}}$

# In[ ]:


def gini(array):
    
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 0.0000001
    array = np.sort(array)
    index = np.arange(1,array.shape[0]+1)
    n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))


# In[ ]:


#This function calculate the gini coefficient for each subject and each sensor (60 seconds information of each sensor)

def sensor_gini(sensor,i):
    return round(gini(np.asarray(All[sensor][i*60:60*(i+1)].round(3).tolist())),3)


# In[ ]:


#creating an empty dataframe with 38186 rows:

statistics_summary = pd.DataFrame(index = range(38186))


# In[ ]:


#calculate gini coefficient and place it in the mentioned dataframe:

Sensors = All.columns[0:13].tolist()
for sensors in Sensors:
    statistics_summary[f'{sensors}_gini'] = ''

for sensors in Sensors:
    gini_ = []
    for i in range(statistics_summary.shape[0]):
        gini_.append(sensor_gini(sensors,i))    
    statistics_summary[f'{sensors}_gini'] = gini_
    
statistics_summary


# # L-moment
# 
# In statistics, L-moments are a sequence of statistics used to summarize the shape of a probability distribution. They are linear combinations of order statistics (L-statistics) analogous to conventional moments, and can be used to calculate quantities analogous to standard deviation, skewness and kurtosis, termed the L-scale, L-skewness and L-kurtosis respectively (the L-mean is identical to the conventional mean). 
# 
# For a random variable X, the rth population L-moment is:
# 
#  $\lambda _{r}=r^{{-1}}\sum _{{k=0}}^{{r-1}}{(-1)^{k}{\binom {r-1}{k}}{\mathrm {E}}X_{{r-k:r}}}$
#  
#  where $X_{k:n}$ denotes the kth order statistic ($k^{th}$ smallest value) in an independent sample of size n from the distribution of X and  $ \mathrm {E}$ denotes expected value. In particular, the first four population L-moments are
# 
# $λ_1 = {\mathrm {E}}X$
# 
# $λ_2 = ( E X_{2 : 2} − E X_{1 : 2} ) / 2 $
# 
# $λ_3 = ( E X_{3 : 3} − 2 E X_{2 : 3} + E X_{1 : 3} ) / 3 $
# 
# $λ_4 = ( E X_{4 : 4} − 3 E X_{3 : 4} + 3 E X_{2 : 4} − E X_{1 : 4} ) / 4$
# 
# Direct estimators for the first four L-moments in a finite sample of n observations are:
# 
# $\ell _{1}={{\tbinom {n}{1}}}^{{-1}}\sum _{{i=1}}^{n}x_{{(i)}}$
# 
# $\ell _{2}={\tfrac {1}{2}}{{\tbinom {n}{2}}}^{{-1}}\sum _{{i=1}}^{n}\left\{{\tbinom {i-1}{1}}-{\tbinom {n-i}{1}}\right\}x_{{(i)}}$
# 
# $\ell _{3}={\tfrac {1}{3}}{{\tbinom {n}{3}}}^{{-1}}\sum _{{i=1}}^{n}\left\{{\tbinom {i-1}{2}}-2{\tbinom {i-1}{1}}{\tbinom {n-i}{1}}+{\tbinom {n-i}{2}}\right\}x_{{(i)}}$
# 
# $\ell _{4}={\tfrac {1}{4}}{{\tbinom {n}{4}}}^{{-1}}\sum _{{i=1}}^{n}\left\{{\tbinom {i-1}{3}}-3{\tbinom {i-1}{2}}{\tbinom {n-i}{1}}+3{\tbinom {i-1}{1}}{\tbinom {n-i}{2}}-{\tbinom {n-i}{3}}\right\}x_{{(i)}}$

# In[ ]:


def L_moment(sensor,i): 
    return lm.lmom_ratios(All[sensor][i*60:60*(i+1)].round(3).tolist(), nmom=5)


# In[ ]:


Sensors = All.columns[0:13].tolist()
for sensors in Sensors:
    statistics_summary[f'{sensors}_L1'] = ''
    statistics_summary[f'{sensors}_L2'] = ''
    statistics_summary[f'{sensors}_L3'] = ''
    statistics_summary[f'{sensors}_L4'] = ''
    statistics_summary[f'{sensors}_L5'] = ''


# In[ ]:


for sensors in Sensors:
    L1 = []
    L2 = []
    L3 = []
    L4 = []
    L5 = []
    
    for i in range(statistics_summary.shape[0]):
        
        L1.append(L_moment(sensors,i)[0])
        L2.append(L_moment(sensors,i)[1])
        L3.append(L_moment(sensors,i)[2])
        L4.append(L_moment(sensors,i)[3])
        L5.append(L_moment(sensors,i)[4])
        
    statistics_summary[f'{sensors}_L1'] = L1    
    statistics_summary[f'{sensors}_L2'] = L2
    statistics_summary[f'{sensors}_L3'] = L3
    statistics_summary[f'{sensors}_L4'] = L4
    statistics_summary[f'{sensors}_L5'] = L5
    
statistics_summary


# # Summary statistics
# 
# In descriptive statistics, summary statistics are used to summarize a set of observations, in order to communicate the largest amount of information as simply as possible.
# 
# Here I try to write some functions for calculating the summary statistics of each subject and each sensor (for 60 seconds information of sensors).
# 

# In[ ]:


def variance(sensor,i):    
    return np.var(All[sensor][i*60:60*(i+1)].round(3).tolist())

def mean_absolute_deviation(sensor,i):    
    return (All[sensor][i*60:60*(i+1)].round(3)).mad()

def IQR(sensor,i):   
    
    Q1 = np.quantile(All[sensor][i*60:60*(i+1)].round(3).tolist(), q=0.2)
    Q3 = np.quantile(All[sensor][i*60:60*(i+1)].round(3).tolist(), q=0.8)
    IQR = Q3 - Q1
    return IQR

def standard_deviation(sensor,i):
    return np.std(All[sensor][i*60:60*(i+1)].round(3).tolist())

def tstd(sensor,i):    
    return stats.tstd(All[sensor][i*60:60*(i+1)].round(3).tolist())

def median_ad(sensor,i):
    return stats.median_abs_deviation(All[sensor][i*60:60*(i+1)].round(3).tolist(), scale="normal")

def Scale(sensor,i):
    t_2 = L_moment(sensors,i)[1] / L_moment(sensors,i)[0]
    return t_2

def Skewness(sensor,i):
    t_3 = L_moment(sensors,i)[2] / L_moment(sensors,i)[1]
    return t_3

def Kurtosis(sensor,i):
    t_4 = L_moment(sensors,i)[3] / L_moment(sensors,i)[2]
    return t_4


# In[ ]:


Sensors = All.columns[0:13].tolist()
for sensors in Sensors:
    statistics_summary[f'{sensors}_variance'] = ''
    statistics_summary[f'{sensors}_mean_ad'] = ''
    statistics_summary[f'{sensors}_IQR'] = ''
    statistics_summary[f'{sensors}_std'] = ''
    statistics_summary[f'{sensors}_tstd'] = ''
    statistics_summary[f'{sensors}_median_ad'] = ''
    statistics_summary[f'{sensors}_Scale'] = ''
    statistics_summary[f'{sensors}_Skewness'] = ''
    statistics_summary[f'{sensors}_Kurtosis'] = ''


# In[ ]:


for sensors in Sensors:
    
    variance_ = []
    mean_ad_ = []
    IQR_ = []
    std_ = []
    tsdt_ = []
    median_ad_ = []
    Scale_ = []
    Skewness_ = []
    Kurtosis_= []
    
    for i in range(statistics_summary.shape[0]):
        
        variance_.append(variance(sensors,i)) 
        mean_ad_.append(mean_absolute_deviation(sensors,i))
        IQR_.append(IQR(sensors,i))
        std_.append(standard_deviation(sensors,i))
        tsdt_.append(tstd(sensors,i))
        median_ad_.append(median_ad(sensors,i))
        Scale_.append(Scale(sensors,i))
        Skewness_.append(Skewness(sensors,i))
        Kurtosis_.append(Kurtosis(sensors,i)) 

    
    statistics_summary[f'{sensors}_variance'] = variance_
    statistics_summary[f'{sensors}_mean_ad'] = mean_ad_
    statistics_summary[f'{sensors}_IQR'] = IQR_
    statistics_summary[f'{sensors}_std'] = std_
    statistics_summary[f'{sensors}_tstd'] = tsdt_
    statistics_summary[f'{sensors}_median_ad'] = median_ad_
    statistics_summary[f'{sensors}_Scale'] = Scale_
    statistics_summary[f'{sensors}_Skewness'] = Skewness_
    statistics_summary[f'{sensors}_Kurtosis'] = Kurtosis_
    
statistics_summary


# In[ ]:


train = statistics_summary[0:25968]
Test = statistics_summary.tail(12218)

train['state'] = ''
train['state'] = labels['state']


# In[ ]:


train


# In[ ]:


statistics_summary


# In[ ]:


train  


# # Correlation 

# In[ ]:


correlation = train.corr(method ='pearson')


# In[ ]:


plt.figure(figsize=(16, 5))
plt.plot(train.columns.tolist(), correlation['state'].tolist())
plt.xlabel("Feature")
plt.ylabel("correlation")
plt.xticks(rotation=90)
plt.show()


# # LGBM Model
# 
# Due to the restriction in computing resource and large size of our dataset, we simply use a LGBM model with just one hyperparameter changed (n_estimators=5000).
# I am sure by tuning this model, we can get a more better accuracy.

# In[ ]:


param = {
       "objective": "binary",
       "metric": "rmse",
       "verbosity": -1,
       "boosting_type": "gbdt",
       "n_estimators": 7000,
       "device": "gpu",
       "gpu_platform_id": 0,
       "gpu_device_id": 0,
   }
   
param2 = { 
       'learning_rate': 0.1,
   }
param.update(param2)




X = train.values[:,:-1]
Y = train.values[:,-1]
label_encoded_y = LabelEncoder().fit_transform(Y)

kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

lgb = LGBMClassifier()

results = cross_val_score(lgb, X, label_encoded_y, cv=kfold)
print(results.mean() ,results.std() )


# In[ ]:


lgb.fit(X, label_encoded_y)
predict_prob = lgb.predict_proba(Test)
predict_prob


# In[ ]:


sample_submission['state'] = predict_prob
sample_submission


# In[ ]:


for i in range(sample_submission.shape[0]):
    sample_submission['state'][i] = 1 - sample_submission['state'][i]
sample_submission


# In[ ]:


sample_submission.to_csv('submission.csv', index=False)


# The above formulas and their interpretation are from wikipedia and the gini coefficient function is from [here](https://github.com/oliviaguest/gini).
