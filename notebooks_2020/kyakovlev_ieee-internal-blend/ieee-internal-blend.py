#!/usr/bin/env python
# coding: utf-8

# # Summary
# 
# I'll try to make a small summary for this blend baseline:
# 
# Step: 0. EDA (missing kernel here, I'll post later)
# 
# 
# Step: 1. Minify Data 
# > https://www.kaggle.com/kyakovlev/ieee-data-minification
# 
# 
# Step: 2. Make ground baseline with no fe:
# > https://www.kaggle.com/kyakovlev/ieee-ground-baseline and 
# > https://www.kaggle.com/kyakovlev/ieee-ground-baseline-deeper-learning
# 
# 
# Step: 3. Make a small FE and see I you can understand data you have
# >  https://www.kaggle.com/kyakovlev/ieee-ground-baseline-make-amount-useful-again and
# >  https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again
# 
# 
# Step: 4. Find good CV strategy 
# >  https://www.kaggle.com/kyakovlev/ieee-cv-options
# and same with gap to compare results (gap in values is what we have in test set)
# https://www.kaggle.com/kyakovlev/ieee-cv-options-with-gap
# 
# Step: 4(1). Groupkfold (by timeblocks) application
# > https://www.kaggle.com/kyakovlev/ieee-lgbm-with-groupkfold-cv
# 
# 
# Step: 5. Try different set of features
# >  https://www.kaggle.com/kyakovlev/ieee-experimental
# 
# 
# Step: 6. Make deeper FE (brute force option)
# > https://www.kaggle.com/kyakovlev/ieee-fe-with-some-eda
# 
# 
# Step: 7. Features selection (missing kernel here, I'll post later)
# 
# 
# Step: 8. Hyperopt (missing kernel here, I'll post later)
# 
# 
# Step: 9. Try other models (XGBoost, CatBoost, NN - missing kernel here, I'll post later)
# > CatBoost (with categorical transformations)  https://www.kaggle.com/kyakovlev/ieee-catboost-baseline-with-groupkfold-cv
# 
# Step: 10. Try blending and stacking (missing kernel here, I'll post later)
# 
# ---
# 
# (Utils)
# 
# Some tricks that where used in fe kernel
# > https://www.kaggle.com/kyakovlev/ieee-small-tricks
# 
# Part of EDA (Just few things)
# > https://www.kaggle.com/kyakovlev/ieee-check-noise and https://www.kaggle.com/kyakovlev/ieee-simple-eda
# 
# ---
# 
# https://www.kaggle.com/c/ieee-fraud-detection/discussion/104142

# In[ ]:


# General imports
import pandas as pd
import os, sys, gc, warnings

warnings.filterwarnings('ignore')


# In[ ]:


########################### DATA LOAD/MIX/EXPORT
#################################################################################
# Simple lgbm (0.0948)
sub_1 = pd.read_csv('../input/ieee-simple-lgbm/submission.csv')

# Blend of two kernels with old features (0.9468)
sub_2 = pd.read_csv('../input/ieee-cv-options/submission.csv')

# Add new features lgbm with CV (0.09485)
sub_3 = pd.read_csv('../input/ieee-lgbm-with-groupkfold-cv/submission.csv')

# Add catboost (0.09407)
sub_4 = pd.read_csv('../input/ieee-catboost-baseline-with-groupkfold-cv/submission.csv')

sub_1['isFraud'] += sub_2['isFraud']
sub_1['isFraud'] += sub_3['isFraud']
sub_1['isFraud'] += sub_4['isFraud']

sub_1.to_csv('submission.csv', index=False)

