#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# **After examining the data, a discrepancy was found in the test and training datasets. Let's try to conduct a comparative analysis of data on the same smartphone model - Samsung Galaxy S20 Ultra**

# **First, read the data. The data in dataset "device_gnss"is more suitable, since the dataset "device_imu" contains data only about the position of the smartphone in space**

# In[ ]:


import pandas as pd
# read the test and train dataset in the Samsung Galaxy S20 Ultra
data2020 = pd.read_csv ('../input/smartphone-decimeter-2022/test/2021-04-28-US-MTV-2/SamsungGalaxyS20Ultra/device_gnss.csv')
data2020.head (10)


# In[ ]:


data2021 = pd.read_csv ('../input/smartphone-decimeter-2022/train/2021-08-04-US-SJC-1/SamsungGalaxyS20Ultra/device_gnss.csv')
data2021.head (10)


# **Clean data in data2020 and data2021**

# In[ ]:


data2020.dtypes


# In[ ]:


data2020.isnull ().sum ()


# In[ ]:


data2021.dtypes


# In[ ]:


data2021.isnull ().sum ()


# In[ ]:


data2020.shape


# In[ ]:


data2021.shape


# **Datasets are almost the same size**

# In[ ]:


data2020.describe ()


# In[ ]:


data2021.describe ()


# **The column "ConstellationType" in data2020 and data2021 gives a hint about the type of satellite. In the future, this will help determine satellites that provide more accurate geodata.**

# In[ ]:


data2020.isnull ().sum (axis = 1).loc [:47]


# In[ ]:


data2021.isnull ().sum (axis = 1).loc [:47]


# **Create dummy columns**

# In[ ]:


data2020 = pd.get_dummies (data2020)
data2020.columns


# In[ ]:


data2021 = pd.get_dummies (data2021)
data2021.columns


# In[ ]:


data2020 = pd.get_dummies (data2020, drop_first = True) 
data2020.columns


# In[ ]:


data2021 = pd.get_dummies (data2021, drop_first = True)
data2021.columns


# **Create a dataframe (X) with the features and a series (y) with the labels**

# In[ ]:


# create a consistent sample from data2020 and data2021
data2020S = data2020.sample (n = 3000)

data2021S = data2021.sample (n = 3000)


# In[ ]:


y = data2020S [['HardwareClockDiscontinuityCount',
       'Svid', 'TimeOffsetNanos', 'State', 'ReceivedSvTimeNanos',
        'AccumulatedDeltaRangeMeters',
       'CarrierFrequencyHz',
       'MultipathIndicator', 'ConstellationType',
       'ChipsetElapsedRealtimeNanos', 'ArrivalTimeNanosSinceGpsEpoch',
       'RawPseudorangeMeters', 'RawPseudorangeUncertaintyMeters',
       'SvPositionXEcefMeters',
       'SvPositionYEcefMeters', 'SvPositionZEcefMeters', 'SvElevationDegrees',
       'SvAzimuthDegrees',
       'SvClockBiasMeters', 'SvClockDriftMetersPerSecond',
       'WlsPositionXEcefMeters', 'WlsPositionYEcefMeters',
       'WlsPositionZEcefMeters', 'CodeType_C', 'CodeType_I', 'CodeType_X',
       'SignalType_BDS_B1I', 'SignalType_GAL_E1', 'SignalType_GAL_E5A',
       'SignalType_GLO_G1', 'SignalType_GPS_L1', 'SignalType_GPS_L5']]

X = data2021S [['HardwareClockDiscontinuityCount',
       'Svid', 'TimeOffsetNanos', 'State', 'ReceivedSvTimeNanos',
        'AccumulatedDeltaRangeMeters',
       'CarrierFrequencyHz',
       'MultipathIndicator', 'ConstellationType',
       'ChipsetElapsedRealtimeNanos', 'ArrivalTimeNanosSinceGpsEpoch',
       'RawPseudorangeMeters', 'RawPseudorangeUncertaintyMeters',
       'SvPositionXEcefMeters',
       'SvPositionYEcefMeters', 'SvPositionZEcefMeters', 'SvElevationDegrees',
       'SvAzimuthDegrees',
       'SvClockBiasMeters', 'SvClockDriftMetersPerSecond',
       'WlsPositionXEcefMeters', 'WlsPositionYEcefMeters',
       'WlsPositionZEcefMeters', 'CodeType_C', 'CodeType_I', 'CodeType_X',
       'SignalType_BDS_B1I', 'SignalType_GAL_E1', 'SignalType_GAL_E5A',
       'SignalType_GLO_G1', 'SignalType_GPS_L1', 'SignalType_GPS_L5']]


# **Divide the dataset into test and train**

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.3, random_state = 42)


# **Exploring data**

# In[ ]:


X.describe ()


# In[ ]:


# Plot the histogram
import matplotlib.pyplot as plt

fig, ax = plt.subplots (figsize = (16, 14))
X.ConstellationType.plot (kind = "hist", ax = ax)


# In[ ]:


# Scatter plot
fig, ax = plt.subplots (figsize = (16, 14))
X.plot.scatter (x = "Svid", y = "RawPseudorangeMeters", ax = ax, alpha = 0.3)

