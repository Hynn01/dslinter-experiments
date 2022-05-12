#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm_notebook as tqdm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# The statistics of training set and test set are very similar.
# 
# However, one thing that caught my eye was the fact that the distribution of the number of unique values (across features) is significantly different between training set and test set.
# 
# It seems that the test set consists of real samples as well as synthetic samples that were generated by sampling the real samples feature distributions (These are probably the "rows which are not included in scoring").
# 
# If this is correct, then finding out which sample is synthetic, and which is real should be relatively easy task:
# 
# Given a sample, we can go over its features and check if the feature value is unique.
# If at least one of the sample's features is unique, then the sample must be a real sample.
# It turns out that if a given sample has no unique values then it is a synthetic sample.
# (It doesn't have to be like that, but in this dataset the probability is seemingly to low that this would not be the case).
# 
# 

# In[ ]:


test_path = '../input/test.csv'

df_test = pd.read_csv(test_path)
df_test.drop(['ID_code'], axis=1, inplace=True)
df_test = df_test.values

unique_samples = []
unique_count = np.zeros_like(df_test)
for feature in tqdm(range(df_test.shape[1])):
    _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)
    unique_count[index_[count_ == 1], feature] += 1

# Samples which have unique values are real the others are fake
real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]

print(len(real_samples_indexes))
print(len(synthetic_samples_indexes))


# 
# If the split between private and public LB sets was done before the resampling process of generating synthetic samples, then it's also possible to regenerate the two different sets.
# For each synthetic sample, we can go over its features and capture those features that have only one instance in the real samples set with the same value, this instance has to be one of the samples' generators.
# 
# 

# In[ ]:


df_test_real = df_test[real_samples_indexes].copy()

generator_for_each_synthetic_sample = []
# Using 20,000 samples should be enough. 
# You can use all of the 100,000 and get the same results (but 5 times slower)
for cur_sample_index in tqdm(synthetic_samples_indexes[:20000]):
    cur_synthetic_sample = df_test[cur_sample_index]
    potential_generators = df_test_real == cur_synthetic_sample

    # A verified generator for a synthetic sample is achieved
    # only if the value of a feature appears only once in the
    # entire real samples set
    features_mask = np.sum(potential_generators, axis=0) == 1
    verified_generators_mask = np.any(potential_generators[:, features_mask], axis=1)
    verified_generators_for_sample = real_samples_indexes[np.argwhere(verified_generators_mask)[:, 0]]
    generator_for_each_synthetic_sample.append(set(verified_generators_for_sample))


# After collecting the "verified generators" for each fake sample, finding the Public/Private LB split is no more than a few set operations.

# In[ ]:


public_LB = generator_for_each_synthetic_sample[0]
for x in tqdm(generator_for_each_synthetic_sample):
    if public_LB.intersection(x):
        public_LB = public_LB.union(x)

private_LB = generator_for_each_synthetic_sample[1]
for x in tqdm(generator_for_each_synthetic_sample):
    if private_LB.intersection(x):
        private_LB = private_LB.union(x)
        
print(len(public_LB))
print(len(private_LB))


# In[ ]:


np.save('public_LB', list(public_LB))
np.save('private_LB', list(private_LB))
np.save('synthetic_samples_indexes', list(synthetic_samples_indexes))

