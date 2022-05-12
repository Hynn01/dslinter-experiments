#!/usr/bin/env python
# coding: utf-8

# In this competition, baseline locations are provided in the ECEF(Earth-Centered Earth-Fixed) coordinate system, so the coordinate system must be converted for submission.

# In[ ]:


import glob
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline

INPUT_PATH = '../input/smartphone-decimeter-2022'

WGS84_SEMI_MAJOR_AXIS = 6378137.0
WGS84_SEMI_MINOR_AXIS = 6356752.314245
WGS84_SQUARED_FIRST_ECCENTRICITY  = 6.69437999013e-3
WGS84_SQUARED_SECOND_ECCENTRICITY = 6.73949674226e-3

HAVERSINE_RADIUS = 6_371_000


# In[ ]:


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


def ecef_to_lat_lng(tripID, gnss_df, UnixTimeMillis):
    ecef_columns = ['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']
    columns = ['utcTimeMillis'] + ecef_columns
    ecef_df = (gnss_df.drop_duplicates(subset='utcTimeMillis')[columns]
               .dropna().reset_index(drop=True))
    ecef = ECEF.from_numpy(ecef_df[ecef_columns].to_numpy())
    blh  = ECEF_to_BLH(ecef)

    TIME = ecef_df['utcTimeMillis'].to_numpy()
    lat = InterpolatedUnivariateSpline(TIME, blh.lat, ext=3)(UnixTimeMillis)
    lng = InterpolatedUnivariateSpline(TIME, blh.lng, ext=3)(UnixTimeMillis)
    return pd.DataFrame({
        'tripId' : tripID,
        'UnixTimeMillis'   : UnixTimeMillis,
        'LatitudeDegrees'  : np.degrees(lat),
        'LongitudeDegrees' : np.degrees(lng),
    })

def calc_score(tripID, pred_df, gt_df):
    d = pandas_haversine_distance(pred_df, gt_df)
    score = np.mean([np.quantile(d, 0.50), np.quantile(d, 0.95)])    
    return score


# In[ ]:


get_ipython().run_cell_magic('capture', '--no-stdout', "\npred_dfs  = []\nscore_list = []\nfor dirname in sorted(glob.glob(f'{INPUT_PATH}/train/*/*')):\n    drive, phone = dirname.split('/')[-2:]\n    tripID  = f'{drive}/{phone}'\n    gnss_df = pd.read_csv(f'{dirname}/device_gnss.csv')\n    gt_df   = pd.read_csv(f'{dirname}/ground_truth.csv')\n    pred_df = ecef_to_lat_lng(tripID, gnss_df, gt_df['UnixTimeMillis'].to_numpy())\n    pred_dfs.append(pred_df)\n    score = calc_score(tripID, pred_df, gt_df)\n    print(f'{tripID:<45}: score = {score:.3f}')\n    score_list.append(score)")


# In[ ]:


baseline_train_df = pd.concat(pred_dfs)
baseline_train_df.to_csv('baseline_train.csv', index=False)


# In[ ]:


mean_score = np.mean(score_list)
print(f'mean_score = {mean_score:.3f}')


# In[ ]:


sample_df = pd.read_csv(f'{INPUT_PATH}/sample_submission.csv')
pred_dfs  = []
for dirname in tqdm(sorted(glob.glob(f'{INPUT_PATH}/test/*/*'))):
    drive, phone = dirname.split('/')[-2:]
    tripID  = f'{drive}/{phone}'
    gnss_df = pd.read_csv(f'{dirname}/device_gnss.csv')
    UnixTimeMillis = sample_df[sample_df['tripId'] == tripID]['UnixTimeMillis'].to_numpy()
    pred_dfs.append(ecef_to_lat_lng(tripID, gnss_df, UnixTimeMillis))
baseline_test_df = pd.concat(pred_dfs)
baseline_test_df.to_csv('baseline_test.csv', index=False)
baseline_test_df.to_csv('submission.csv', index=False)

