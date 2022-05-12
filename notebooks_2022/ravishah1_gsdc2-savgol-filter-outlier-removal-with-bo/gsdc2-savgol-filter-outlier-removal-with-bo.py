#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# This notebook is designed to give you an introduction on how to approach this competition and use gnss data. I perform outlier correction and apply a savgol filter after hyperparameter tuning with bayesian optimization.
# 
# This notebook is broken down into a few sections. 
# 1. Standard Functions and Constants - this code is mostly helper functions borrowed from the notebook by @saitodevel01. It is used to generate the baseline which I have included as a datasource to save time. This section also contains my imports and evaluation function.
# 2. Outlier Correction - here I detect outliers by comparing the lat and lon at each timestep to the timestep before and after. If the haversine distance between the points is greater than a threshold, it is flagged as an outlier. I then replace outliers with the mean of the lat and lon at the previous and future timestep.
# 3. Savgol Filter - here I have defined a function to apply scipy’s savgol filter algorithm to the lat and lon columns. The function is set up to hyperparameter tune the window length and poly order. 
# 4. Bayesian Optimization - here I use skopt’s gp_minimize function in order to apply Bayesian optimization using Gaussian Processes. I optimize the outlier correction threshold, savgol filter window length, and savgol filter poly order
# 5. Submit - uses optimal parameters to generate submission file
# 
# **References**
# 
# https://www.kaggle.com/code/saitodevel01/gsdc2-baseline-submission by @saitodevel01 - used for baseline generation
# 
# https://www.kaggle.com/code/dehokanta/baseline-post-processing-by-outlier-correction by @dehokanta - notebook from last year’s competition inspired outlier correction technique
# 
# https://www.kaggle.com/code/tqa236/kalman-filter-hyperparameter-search-with-bo by @tqa236 - notebook from last year’s competition inspiration for bayesian optimization

# # Standard Functions and Constants

# In[ ]:


import glob
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from pathlib import Path

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import savgol_filter

from skopt import gp_minimize
from skopt.space import Real, Integer

import warnings
warnings.filterwarnings('ignore')

INPUT_PATH = '../input/smartphone-decimeter-2022'
bl_path = '../input/gsdc2-baseline-submission'
bl_train = pd.read_csv(f'{bl_path}/baseline_train.csv')
bl_test = pd.read_csv(f'{bl_path}/baseline_test.csv')


# In[ ]:


WGS84_SEMI_MAJOR_AXIS = 6378137.0
WGS84_SEMI_MINOR_AXIS = 6356752.314245
WGS84_SQUARED_FIRST_ECCENTRICITY  = 6.69437999013e-3
WGS84_SQUARED_SECOND_ECCENTRICITY = 6.73949674226e-3

HAVERSINE_RADIUS = 6_371_000


# In[ ]:


# reference https://www.kaggle.com/code/saitodevel01/gsdc2-baseline-submission

@dataclass
class ECEF:
    x: np.array
    y: np.array
    z: np.array

    def to_numpy(self):
        return np.stack([self.x, self.y, self.z], axis=0)

    @staticmethod
    def from_numpy(pos):
        x, y, z = [np.squeeze(w) for w in np.split(pos, 3, axis=-1)]
        return ECEF(x=x, y=y, z=z)

@dataclass
class BLH:
    lat : np.array
    lng : np.array
    hgt : np.array

def ECEF_to_BLH(ecef):
    a = WGS84_SEMI_MAJOR_AXIS
    b = WGS84_SEMI_MINOR_AXIS
    e2  = WGS84_SQUARED_FIRST_ECCENTRICITY
    e2_ = WGS84_SQUARED_SECOND_ECCENTRICITY
    x = ecef.x
    y = ecef.y
    z = ecef.z
    r = np.sqrt(x**2 + y**2)
    t = np.arctan2(z * (a/b), r)
    B = np.arctan2(z + (e2_*b)*np.sin(t)**3, r - (e2*a)*np.cos(t)**3)
    L = np.arctan2(y, x)
    n = a / np.sqrt(1 - e2*np.sin(B)**2)
    H = (r / np.cos(B)) - n
    return BLH(lat=B, lng=L, hgt=H)

def haversine_distance(blh_1, blh_2):
    dlat = blh_2.lat - blh_1.lat
    dlng = blh_2.lng - blh_1.lng
    a = np.sin(dlat/2)**2 + np.cos(blh_1.lat) * np.cos(blh_2.lat) * np.sin(dlng/2)**2
    dist = 2 * HAVERSINE_RADIUS * np.arcsin(np.sqrt(a))
    return dist

def pandas_haversine_distance(df1, df2):
    blh1 = BLH(
        lat=np.deg2rad(df1['LatitudeDegrees'].to_numpy()),
        lng=np.deg2rad(df1['LongitudeDegrees'].to_numpy()),
        hgt=0,
    )
    blh2 = BLH(
        lat=np.deg2rad(df2['LatitudeDegrees'].to_numpy()),
        lng=np.deg2rad(df2['LongitudeDegrees'].to_numpy()),
        hgt=0,
    )
    return haversine_distance(blh1, blh2)


# In[ ]:


def calc_score(tripID, pred_df, gt_df):
    d = pandas_haversine_distance(pred_df, gt_df)
    score = np.mean([np.quantile(d, 0.50), np.quantile(d, 0.95)])    
    return score


# # Outlier Correction

# In[ ]:


def calc_haversine(lat1, lon1, lat2, lon2):
    """Calculates the great circle distance between two points
    on the earth. Inputs are array-like and specified in decimal degrees.
    """
    RADIUS = 6_367_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 +         np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    dist = 2 * RADIUS * np.arcsin(a**0.5)
    return dist

def correct_outliers(df, th=2):
    df['dist_pre'] = 0
    df['dist_pro'] = 0

    df['latDeg_pre'] = df['LatitudeDegrees'].shift(periods=1,fill_value=0)
    df['lngDeg_pre'] = df['LongitudeDegrees'].shift(periods=1,fill_value=0)
    df['latDeg_pro'] = df['LatitudeDegrees'].shift(periods=-1,fill_value=0)
    df['lngDeg_pro'] = df['LongitudeDegrees'].shift(periods=-1,fill_value=0)
    df['dist_pre'] = calc_haversine(df.latDeg_pre, df.lngDeg_pre, df.LatitudeDegrees, df.LongitudeDegrees)
    df['dist_pro'] = calc_haversine(df.LatitudeDegrees, df.LongitudeDegrees, df.latDeg_pro, df.lngDeg_pro)

    df.loc[df.index.min(), 'dist_pre'] = 0
    df.loc[df.index.max(), 'dist_pro'] = 0
    
    pro_95 = df['dist_pro'].mean() + (df['dist_pro'].std() * th)
    pre_95 = df['dist_pre'].mean() + (df['dist_pre'].std() * th)

    ind = df[(df['dist_pro'] > pro_95)&(df['dist_pre'] > pre_95)][['dist_pre','dist_pro']].index

    for i in ind:
        df.loc[i,'LatitudeDegrees'] = (df.loc[i-1,'LatitudeDegrees'] + df.loc[i+1,'LatitudeDegrees'])/2
        df.loc[i,'LongitudeDegrees'] = (df.loc[i-1,'LongitudeDegrees'] + df.loc[i+1,'LongitudeDegrees'])/2
    
    return df


# # Savgol Filter

# In[ ]:


def apply_savgol_filter(df, wl, poly):
    df.LatitudeDegrees = savgol_filter(df.LatitudeDegrees, wl, poly)
    df.LongitudeDegrees = savgol_filter(df.LongitudeDegrees, wl, poly)
    return df


# # Bayesian Optimization

# In[ ]:


def optimize(params):
    th, wl, poly = params
    if wl%2==0:
        wl+=1
    
    score_list = []

    for tripID in sorted(bl_train.tripId.unique()):

        gt_df   = pd.read_csv(f'{INPUT_PATH}/train/{tripID}/ground_truth.csv')
        pred_df = bl_train[bl_train.tripId == tripID]

        pred_df = correct_outliers(pred_df, th)
        pred_df = apply_savgol_filter(pred_df, wl, poly)

        score = calc_score(tripID, pred_df, gt_df)
        score_list.append(score)

    mean_score = np.mean(score_list)
    return mean_score


# In[ ]:


space = [Real(1.5, 2.5, name='threshhold'), 
         Integer(7, 31, name='window_len'), 
         Integer(2, 6, name='poly_order')]

result = gp_minimize(optimize, space, n_calls=100)


# In[ ]:


print(f'best train score: {result.fun}')


# In[ ]:


if result.x[1]%2==0:
    result.x[1]+=1

print(f'best params:\noutlier threshhold: {result.x[0]}\nsavgol filter window length: {result.x[1]}\nsavgol filter poly order: {result.x[2]}')


# # Submit

# In[ ]:


preds = list()

for tripID in sorted(bl_test.tripId.unique()):
    pred_df = bl_test[bl_test.tripId == tripID]

    pred_df = correct_outliers(pred_df, result.x[0])
    pred_df = apply_savgol_filter(pred_df, result.x[1], result.x[2])

    preds.append(pred_df)
    
sub = pd.concat(preds)
sub = sub[["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]]
sub.to_csv('submission.csv', index=False)


# In[ ]:




