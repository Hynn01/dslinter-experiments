#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# packages

# standard
import numpy as np
import pandas as pd
import time

# plots
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning tools
import h2o
from h2o.estimators import H2OGeneralizedLinearEstimator, H2ORandomForestEstimator, H2OGradientBoostingEstimator


# # Import and First Glance

# In[ ]:


# load data + first glance
t1 = time.time()
df_train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')
df_test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv')
df_sub = pd.read_csv('../input/tabular-playground-series-may-2022/sample_submission.csv')
t2 = time.time()
print('Elapsed time:', np.round(t2-t1,4))


# In[ ]:


# first glance (training data)
df_train.head()


# In[ ]:


# dimensions
print('Train Set:', df_train.shape)
print('Test Set :', df_test.shape)


# In[ ]:


# structure / missing values
df_train.info(verbose=True, show_counts=True)


# In[ ]:


# same for test set
df_test.info(verbose=True, show_counts=True)


# # Features

# In[ ]:


# f27 is special...
df_train.f_27.value_counts()


# In[ ]:


# aux function
def extract_char(i_string, i_k):
    return i_string[i_k]


# In[ ]:


# decompose f_27 in character features
for k in range(10):
    feature_name = 'f_27_' + str(k)
    print(feature_name)
    df_train[feature_name] = list(map(lambda x: extract_char(x,k), df_train.f_27))
    df_test[feature_name] = list(map(lambda x: extract_char(x,k), df_test.f_27))


# In[ ]:


df_train['unique_chars'] = df_train.f_27.apply(lambda s: len(set(s)))
df_test['unique_chars'] = df_test.f_27.apply(lambda s: len(set(s)))


# In[ ]:


features_num = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 
                'f_06', 'f_07', 'f_08', 'f_09', 'f_10', 'f_11',
                'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17',
                'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23',
                'f_24', 'f_25', 'f_26', 'f_28', 'f_29', 'f_30']


# In[ ]:


features_char = ['f_27_0', 'f_27_1', 'f_27_2', 'f_27_3', 'f_27_4', 
                 'f_27_5', 'f_27_6', 'f_27_7', 'f_27_8', 'f_27_9',
                 'unique_chars']


# In[ ]:


# numerical features
df_train[features_num].describe()


# In[ ]:


df_train[features_char].describe(include='all')


# In[ ]:


# distribution of each character feature
for f in features_char:
    plt.figure(figsize=(10,3))
    df_train[f].value_counts().sort_index().plot(kind='bar')
    plt.title(f + ' - Train')
    plt.grid()
    plt.show()


# # Target

# In[ ]:


# target - basic stats
print(df_train.target.value_counts())
df_train.target.value_counts().plot(kind='bar')
plt.title('Target')
plt.grid()
plt.show()


# In[ ]:


# plot each numerical feature split by target=0/1
for f in features_num:
    plt.figure(figsize=(10,3))
    sns.violinplot(data=df_train, y='target', x=f, orient='h')
    plt.title(f + ' - Train')
    plt.grid()
    plt.show()


# In[ ]:


for f in features_char:
    ctab = pd.crosstab(df_train[f], df_train.target)
    ctab_norm = ctab.transpose() / (ctab.sum(axis=1))
    plt.figure(figsize=(16,3))
    sns.heatmap(ctab_norm, annot=True, 
                cmap='Blues',
                linecolor='black',
                linewidths=0.1)
    plt.title(f)
    plt.show()


# # Fit Model

# In[ ]:


# select predictors
predictors = features_num + features_char
print('Number of predictors: ', len(predictors))
print(predictors)


# In[ ]:


# start H2O
h2o.init(max_mem_size='12G', nthreads=4) # Use maximum of 12 GB RAM and 4 cores


# In[ ]:


# upload train/test set to H2O environment
t1 = time.time()
train_hex = h2o.H2OFrame(df_train)
test_hex = h2o.H2OFrame(df_test)
t2 = time.time()
print('Elapsed time [s]: ', np.round(t2-t1,2))

# force categorical target
train_hex['target'] = train_hex['target'].asfactor()


# In[ ]:


#  fit Gradient Boosting model
n_cv = 5

fit_GBM = H2OGradientBoostingEstimator(ntrees=750,
                                       max_depth=15,
                                       min_rows=10,
                                       learn_rate=0.1, # default: 0.1
                                       sample_rate=1,
                                       col_sample_rate=0.5,
                                       nfolds=n_cv,
                                       score_each_iteration=True,
                                       stopping_metric='auc',
                                       stopping_rounds=5,
                                       stopping_tolerance=0.00001,
                                       seed=999)
# train model
t1 = time.time()
fit_GBM.train(x=predictors,
              y='target',
              training_frame=train_hex)
t2 = time.time()
print('Elapsed time [s]: ', np.round(t2-t1,2))


# In[ ]:


# show cross validation metrics
fit_GBM.cross_validation_metrics_summary()


# In[ ]:


# show scoring history - training vs cross validations
for i in range(n_cv):
    cv_model_temp = fit_GBM.cross_validation_models()[i]
    df_cv_score_history = cv_model_temp.score_history()
    my_title = 'CV ' + str(1+i) + ' - Scoring History [AUC]'
    plt.scatter(df_cv_score_history.number_of_trees,
                y=df_cv_score_history.training_auc, 
                c='blue', label='training')
    plt.scatter(df_cv_score_history.number_of_trees,
                y=df_cv_score_history.validation_auc, 
                c='darkorange', label='validation')
    plt.title(my_title)
    plt.xlabel('Number of Trees')
    plt.ylabel('AUC')
    plt.ylim(0.7,1.0)
    plt.legend()
    plt.grid()
    plt.show()


# In[ ]:


# variable importance
fit_GBM.varimp_plot(30)
plt.show()


# In[ ]:


# training performance
perf_train = fit_GBM.model_performance(train=True)
perf_train.plot()
plt.show()


# In[ ]:


# cross validation performance
perf_cv = fit_GBM.model_performance(xval=True)
perf_cv.plot()
plt.show()


# In[ ]:


# predict on train set (extract probabilities only)
pred_train_GBM = fit_GBM.predict(train_hex)['p1']
pred_train_GBM = pred_train_GBM.as_data_frame().p1

# plot train set predictions (probabilities)
plt.figure(figsize=(8,4))
plt.hist(pred_train_GBM, bins=100)
plt.title('Predictions on Train Set - GBM')
plt.grid()
plt.show()


# In[ ]:


# check calibration
n_actual = sum(df_train.target)
n_pred_GBM = sum(pred_train_GBM)

print('Actual Frequency    :', n_actual)
print('Predicted Frequency :', n_pred_GBM)
print('Calibration Ratio   :', n_pred_GBM / n_actual)


# In[ ]:


# predict on test set (extract probabilities only)
pred_test_GBM = fit_GBM.predict(test_hex)['p1']
pred_test_GBM = pred_test_GBM.as_data_frame().p1


# In[ ]:


# plot test set predictions (probabilities)
plt.figure(figsize=(8,4))
plt.hist(pred_test_GBM, bins=100)
plt.title('Predictions on Test Set - GBM')
plt.grid()
plt.show()


# In[ ]:


# GBM submission
df_sub_GBM = df_sub.copy()
df_sub_GBM.target = pred_test_GBM
display(df_sub_GBM.head())
# save to file
df_sub_GBM.to_csv('submission_GBM.csv', index=False)

