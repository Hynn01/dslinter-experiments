#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# <img src="https://i.imgur.com/SqVoZ5w.jpg" width="400px"> 

# Welcome to the "University of Liverpool - Ion Switching" competition! In this competition, contestants are challenged to predict the number of open ion channels based on electrophysiological signals from human cells. This is an important problem because potential solutions can have far-reaching impacts. From human diseases to how climate change affects plants, faster detection of ion channels could greatly accelerate solutions to major world problems.
# 
# In this kernel, I will briefly explain the background knowledge required to understand the dataset. Then, I will visualize the data with Plotly and Matplotlib.
# 
# <font size=3 color="red">Please upvote this kernel if you like it. It motivates me to produce more quality content :)</font>

# # Contents
# 
# * [<font size=4>Background</font>](#1)
#     * [Electrophysiology](#1.1)
#     * [Clamp method](#1.2)
#     * [Ion channels](#1.3)
# 
# 
# * [<font size=4>EDA</font>](#2)
#     * [Preparing the ground](#2.1)
#     * [Signal data](#2.2)
#     * [Denoising](#2.3)
#     * [Open channels](#2.4)
#     * [Measures of complexity (entropy and fractal dimension)](#2.5)
# 
# 
# * [<font size=4>Ending Note</font>](#3)

# # Acknowledgements
# 
# 1. [EDA - Ion Switching ~ by Peter](https://www.kaggle.com/pestipeti/eda-ion-switching)
# 2. [Simple Ion Ridge Regression Starter ~ by Bojan](https://www.kaggle.com/tunguz/simple-ion-ridge-regression-starter)
# 3. [Electrophysiology ~ by Wikipedia](https://en.wikipedia.org/wiki/Electrophysiology)

# # Background <a id="1"></a>
# 
# ## Electrophysiology <a id="1.1"></a>
# 
# Electrophysiology is a branch of physiology where the electrical properties of biological cells and tissues are studied. It involves the measurement of voltage or current changes on a wide variety of scales from single ion channel proteins to whole organs like the heart. In neuroscience, it includes the measurement of electrical activity in neurons, and, in particular, action potential activity. These measured signals are then used to perform medical diagnoses and analyses on humans. Below is an example of an elecrophysiological signal:
# 
# <img src="https://i.imgur.com/9X3V7uw.gif" width="500px">
# 
# From the above graph, it can be seen that the voltage is being plotted against time. These voltages are measured from human organs using special methods and apparatus. One such method is called the **clamp method**.

# ## Clamp method <a id="1.2"></a>
# 
# The clamp method is one of many methods to measure voltage and current changes in human organs. Below is a video that explains how the clamp method works. **The relevant section ends at 2:55**.

# In[ ]:


from IPython.display import HTML
HTML('<center><iframe width="700" height="400" src="https://www.youtube.com/embed/CvfXjGVNbmw?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe></center>')


# The clamp method uses a pair of electrodes, an amplifier, and a signal generator to measure voltage and current changes in organs. A simple diagram of the setup is given below:
# 
# <img src="https://i.imgur.com/SNMuuuf.png" width="400px">
# 
# The electrodes in the aobve diagram measure the electrical impulse from the axon (a part of the neuron). These electric signals are then amplified by the potential amplifier because these electric signals are very small in amplitude. Without amplification, these signals would go undetected. These signals are fianlly displayed on a screen using a signal generator and a montior (display screen). Below is a picture of an actual clamp method setup:

# ## Ion channels <a id="1.3"></a>
# 
# Now since we understand how voltage and current signals are measured and recorded, let us understand the meaning of "ion channels". 
# 
# Ion channels are pore-forming membrane proteins that allow ions to pass through the channel pore. Ion channels are "closed" when they do not allow ions to pass through and "open" when they allow ions to pass through. Ion channels are especially prominent components of the nervous system. In addition, they are key components in a wide variety of biological processes that involve rapid changes in cells, such as cardiac, skeletal, and smooth muscle contraction, epithelial transport of nutrients and ions, T-cell activation and pancreatic beta-cell insulin release.
# 
# **The pivotal role of ion channels in several biological processes makes it an excellent way to discover new drugs and medicines for various diseases.**

# ### Therefore, finding a relationship between current signals and open ion channels can unlock new possibilities in the fields of medicine and environmental studies. And hence this competition.

# # EDA <a id="2"></a>
# 
# Now, I will try to visualize the data and gain some insights from it.

# ## Preparing the ground <a id="2.1"></a>

# ### Import libraries

# In[ ]:


import os
import gc
import time
import math
from numba import jit
from math import log, floor
from sklearn.neighbors import KDTree

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle

import seaborn as sns
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import pywt 
from statsmodels.robust import mad

import scipy
from scipy import signal
from scipy.signal import butter, deconvolve


# ### Define hyperparameters and file paths

# In[ ]:


SAMPLE_RATE = 25
SIGNAL_LEN = 1000

TEST_PATH = "../input/liverpool-ion-switching/test.csv"
TRAIN_PATH = "../input/liverpool-ion-switching/train.csv"
SUBMISSION_PATH = "../input/liverpool-ion-switching/sample_submission.csv"


# ### Load the data

# In[ ]:


test_data = pd.read_csv(TEST_PATH)
train_data = pd.read_csv(TRAIN_PATH)
test_data.drop(columns=['time'], inplace=True)


# ## Signal data <a id="2.2"></a>

# ### Signal data vs. Time

# In[ ]:


plt.figure(figsize=(20, 10))
plt.plot(train_data["time"], train_data["signal"], color="r")
plt.title("Signal data", fontsize=20)
plt.xlabel("Time", fontsize=18)
plt.ylabel("Signal", fontsize=18)
plt.show()


# In the graph above, we can see how the signal (current) changes over time. The signal seems to have a simple oscillatory nature until time *t = 300*. The current rapidly oscialltes up and down forming a sort of "rectangular" shape when viewed at scale. After *t = 300*, however, the current assumes a different shape. The signal starts oscillating at the macroscopic level, displaying a trend similar to *y = sin<sup>2</sup>(x)*, but with monotonously increasing amplitudes. This change in trend has an effect on the open ion channels, as we will see later.

# ### Sample signal snippets

# In[ ]:


fig = make_subplots(rows=3, cols=1)

x_1 = train_data.loc[:100]["time"]
y_1 = train_data.loc[:100]["signal"]
x_2 = train_data.loc[100:200]["time"]
y_2 = train_data.loc[100:200]["signal"]
x_3 = train_data.loc[200:300]["time"]
y_3 = train_data.loc[200:300]["signal"]

fig.add_trace(go.Scatter(x=x_1, y=y_1, showlegend=False,
                    mode='lines+markers', name="First sample",
                         marker=dict(color="dodgerblue")),
             row=1, col=1)

fig.add_trace(go.Scatter(x=x_2, y=y_2, showlegend=False,
                    mode='lines+markers', name="Second sample",
                         marker=dict(color="mediumseagreen")),
             row=2, col=1)

fig.add_trace(go.Scatter(x=x_3, y=y_3, showlegend=False,
                    mode='lines+markers', name="Third sample",
                         marker=dict(color="violet")),
             row=3, col=1)

fig.update_layout(height=1200, width=800, title_text="Sample signals")
fig.show()


# Above, I have plotted three snippets from the current signal to give an idea of the volatility in the time series.

# ## Denoising <a id="2.3"></a>
# 
# Now, I will show these volatile signals can be denoised in order to extract the underlying trend from the signal. This method may lose some information from the original signal, but it may be useful in extracting certain features regarding the trends in the time series.

# ### Wavelet denoising
# 
# Wavelet denoising is a way to remove the unnecessary noise from a signal. This method calculates coefficients called the "wavelet coefficients". These coefficients decide which pieces of information to keep (signal) and which ones to discard (noise).
# 
# We make use of the MAD value (mean absolute deviation) to understand the randomness in the signal and accordingly decide the minimum threshold for the wavelet coefficients in the time series. We filter out the low coefficients from the wavelet coefficients and reconstruct the electric signal from the remaining coefficients and that's it; we have successfully removed noise from the electric signal.

# In[ ]:


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise_signal(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    return pywt.waverec(coeff, wavelet, mode='per')


# In[ ]:


x = train_data.loc[:100]["time"]
y1 = train_data.loc[:100]["signal"]
y_w1 = denoise_signal(train_data.loc[:100]["signal"])
y2 = train_data.loc[100:200]["signal"]
y_w2 = denoise_signal(train_data.loc[100:200]["signal"])
y3 = train_data.loc[200:300]["signal"]
y_w3 = denoise_signal(train_data.loc[200:300]["signal"])


# In[ ]:


fig = make_subplots(rows=3, cols=1)

fig.add_trace(
    go.Scatter(x=x, mode='lines+markers', y=y1, marker=dict(color="lightskyblue"), showlegend=False,
               name="Original signal"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=x, y=y_w1, mode='lines', marker=dict(color="navy"), showlegend=False,
               name="Denoised signal"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=x, mode='lines+markers', y=y2, marker=dict(color="mediumaquamarine"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=x, y=y_w2, mode='lines', marker=dict(color="darkgreen"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=x, mode='lines+markers', y=y3, marker=dict(color="thistle"), showlegend=False),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=x, y=y_w3, mode='lines', marker=dict(color="indigo"), showlegend=False),
    row=3, col=1
)

fig.update_layout(height=1200, width=800, title_text="Original (pale) vs. Denoised (dark) signals")
fig.show()


# In the above graphs, the dark lineplots represent the denoised signals and the light lineplots represent the original signals. We can see that the wavelet denoising method is able to successfully capture the trend in the signal, while ignoring the noise at the same time. These trends can be used to calculate useful features for modeling.

# The below graphs illustrate these graphs side-by-side. Red graphs represent original signals and green graphs represent denoised signals.

# In[ ]:


x = train_data.loc[:100]["time"]
y1 = train_data.loc[:100]["signal"]
y_w1 = denoise_signal(train_data.loc[:100]["signal"])
y2 = train_data.loc[100:200]["signal"]
y_w2 = denoise_signal(train_data.loc[100:200]["signal"])
y3 = train_data.loc[200:300]["signal"]
y_w3 = denoise_signal(train_data.loc[200:300]["signal"])

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(30, 20))

ax[0, 0].plot(y1, color='seagreen', marker='o') 
ax[0, 0].set_title('Original Signal', fontsize=24)
ax[0, 1].plot(y_w1, color='red', marker='.') 
ax[0, 1].set_title('After Wavelet Denoising', fontsize=24)

ax[1, 0].plot(y2, color='seagreen', marker='o') 
ax[1, 0].set_title('Original Signal', fontsize=24)
ax[1, 1].plot(y_w2, color='red', marker='.') 
ax[1, 1].set_title('After Wavelet Denoising', fontsize=24)

ax[2, 0].plot(y3, color='seagreen', marker='o') 
ax[2, 0].set_title('Original Signal', fontsize=24)
ax[2, 1].plot(y_w3, color='red', marker='.') 
ax[2, 1].set_title('After Wavelet Denoising', fontsize=24)

plt.show()


# ### Average smoothing
# 
# Average smooting is a relatively simple way to denoise signals. In this method, we take a "window" with a fixed size (like 10). We first place the window at the beginning of the time series (first ten elements) and calculate the mean of that section. We now move the window across the time series in the forward direction by a particular "stride", calculate the mean of the new window and repeat the process, until we reach the end of the time series. All the mean values we calculated are concatenated into a new time series, which forms the denoised signal.

# In[ ]:


def average_smoothing(signal, kernel_size=3, stride=1):
    sample = []
    start = 0
    end = kernel_size
    while end <= len(signal):
        start = start + stride
        end = end + stride
        sample.extend(np.ones(end - start)*np.mean(signal[start:end]))
    return np.array(sample)


# In[ ]:


x = train_data.loc[:100]["time"]
y1 = train_data.loc[:100]["signal"]
y_a1 = average_smoothing(train_data.loc[:100]["signal"])
y2 = train_data.loc[100:200]["signal"]
y_a2 = average_smoothing(train_data.loc[100:200]["signal"])
y3 = train_data.loc[200:300]["signal"]
y_a3 = average_smoothing(train_data.loc[200:300]["signal"])


# In[ ]:


fig = make_subplots(rows=3, cols=1)

fig.add_trace(
    go.Scatter(x=x, mode='lines+markers', y=y1, marker=dict(color="lightskyblue"), showlegend=False,
               name="Original signal"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=x, y=y_a1, mode='lines', marker=dict(color="navy"), showlegend=False,
               name="Denoised signal"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=x, mode='lines+markers', y=y2, marker=dict(color="mediumaquamarine"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=x, y=y_a2, mode='lines', marker=dict(color="darkgreen"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=x, mode='lines+markers', y=y3, marker=dict(color="thistle"), showlegend=False),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=x, y=y_a3, mode='lines', marker=dict(color="indigo"), showlegend=False),
    row=3, col=1
)

fig.update_layout(height=1200, width=800, title_text="Original (pale) vs. Denoised (dark) signals")
fig.show()


# In the above graphs, the dark lineplots represent the denoised signals and the light lineplots represent the original signals. We can see that the average smoothing method is not able to effectively remove the noise and uncover the trend. A lot of the noise in the original signal persists even after denoising. Therefore, wavelet denoising is clearly more effective at finding trends in the electric signals.

# The below graphs illustrate these graphs side-by-side. Red graphs represent original signals and green graphs represent denoised signals.

# In[ ]:


x = train_data.loc[:100]["time"]
y1 = train_data.loc[:100]["signal"]
y_a1 = average_smoothing(train_data.loc[:100]["signal"])
y2 = train_data.loc[100:200]["signal"]
y_a2 = average_smoothing(train_data.loc[100:200]["signal"])
y3 = train_data.loc[200:300]["signal"]
y_a3 = average_smoothing(train_data.loc[200:300]["signal"])

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(30, 20))

ax[0, 0].plot(y1, color='seagreen', marker='o') 
ax[0, 0].set_title('Original Signal', fontsize=24)
ax[0, 1].plot(y_a1, color='red', marker='.') 
ax[0, 1].set_title('After Average Smoothing', fontsize=24)

ax[1, 0].plot(y2, color='seagreen', marker='o') 
ax[1, 0].set_title('Original Signal', fontsize=24)
ax[1, 1].plot(y_a2, color='red', marker='.') 
ax[1, 1].set_title('After Average Smoothing', fontsize=24)

ax[2, 0].plot(y3, color='seagreen', marker='o') 
ax[2, 0].set_title('Original Signal', fontsize=24)
ax[2, 1].plot(y_a3, color='red', marker='.') 
ax[2, 1].set_title('After Average Smoothing', fontsize=24)

plt.show()


# ## Open channels <a id="2.4"></a>

# In[ ]:


signals = []
targets = []

train = train_data # shuffle(train_data).reset_index(drop=True)
for i in range(4000):
    min_lim = SIGNAL_LEN * i
    max_lim = SIGNAL_LEN * (i + 1)
    
    signals.append(list(train["signal"][min_lim : max_lim]))
    targets.append(train["open_channels"][max_lim])
    
signals = np.array(signals)
targets = np.array(targets)


# In[ ]:


plt.figure(figsize=(20, 10))
plt.plot(train_data["time"], train_data["open_channels"], color="b")
plt.title("Open channels", fontsize=20)
plt.xlabel("Time", fontsize=18)
plt.ylabel("Channels", fontsize=18)
plt.show()


# The above graph contains all the open channels values in the train dataset. The number of open channels seem to be limited to 0 or 1 until *t = 150*. However, after that, we can see the number of open channels taking on a muchlarger range of values until *t = 500*. This increase in range and variance of *open_channels* seems to coincide with the increases volatility in the *signal* after *t = 200*. We can infer that an increase in electric activity results in greater variance in the number of open channels itself.

# ### Open channels distribution

# In[ ]:


fig = go.Figure(data=[
    go.Bar(x=list(range(11)), y=train_data['open_channels'].value_counts(sort=False).values)
])

fig.update_layout(title='Target (open_channels) distribution')
fig.show()


# From the above graph, we can see that *open_channels* has a discrete probability distribution, with values ranging from 0 to 10 occuring in the training data. The proability generally seems to decrease exponentially with increase in *open_channels*. THis means that a high number of open channels is more unlikely than a smaller number of open channels. Maybe, a closed state is a more stable than an open state for an ion channel, and so it is more likely to have a lower number of open channels.

# ### Open channels distribution for different batches

# In[ ]:


fig = make_subplots(rows=5, cols=2, subplot_titles=["Batch #{}".format(i) for i in range(10)])
i = 0

for row in range(1, 6):
    for col in range(1, 3):   
        
        data = train_data.iloc[(i * 500000):((i+1) * 500000 + 1)]['open_channels'].value_counts(sort=False).values
        fig.add_trace(go.Bar(x=list(range(11)), y=data), row=row, col=col)
        
        i += 1

fig.update_layout(title_text="Target distribution in different batches", height=1200, showlegend=False)
fig.show()


# From the graphs above, we can see that *open_channels* has different distributions for different batch numbers. The greater the batch number, the greater the variance and range of *open_channels*. For example, batch 0 only contains the values 0 and 1, whereas, batch 9 has 8 different values ranging from 0 to 7.

# ### Signal mean vs. Open channels

# In[ ]:


df = pd.DataFrame(np.transpose([np.mean(np.abs(signals), axis=1), targets]))
df.columns = ["signal_mean", "open_channels"]
fig = go.Figure()

channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for channel in channels:
    fig.add_trace(go.Box(x=df['open_channels'][df['open_channels'] == channel],
                         y=df['signal_mean'][df['open_channels'] == channel],
                         name=channel,
                         marker=dict(color='seagreen'), showlegend=False)
                         )
    
fig.add_trace(go.Scatter(x=channels,
                         y=[df['signal_mean'][df['open_channels'] == channel].median() for channel in channels],
                         mode="lines+markers",
                         name=channel,
                         marker=dict(color='seagreen'), showlegend=False)
                         )

fig.update_layout(title="Signal mean vs. Open channels", xaxis_title="Open channels", yaxis_title="Signal mean")
fig.show()


# In[ ]:


df = pd.DataFrame(np.transpose([np.mean(np.abs(signals), axis=1), targets]))
df.columns = ["signal_mean", "open_channels"]
fig = go.Figure()

channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

fig.add_trace(go.Scatter(x=channels,
                         y=[df['signal_mean'][df['open_channels'] == channel].median() for channel in channels],
                         mode="lines+markers",
                         name=channel,
                         marker=dict(color='seagreen'), showlegend=False)
                         )

fig.update_layout(title="Median signal mean vs. Open channels", xaxis_title="Open channels", yaxis_title="Median signal mean")
fig.show()


# From the above graph, we can see that the signal mean is between 1.5 and 3.5 for almost all values of *open_channels*. But, there are clear differences in the distributions of signal mean. Most distributions are approximately normal (centered at 2.95). In general, the signal seems to be greater for greater values of *open_channels* with the exceptions of *open_channels = 1* or *2*.

# ### Signal data vs. Open channels

# In[ ]:


plt.figure(figsize=(20, 10))
plt.plot(train_data["time"], train_data["signal"], color="r")
plt.plot(train_data["time"], train_data["open_channels"], color="b")
plt.title("Signal data vs. Open channels", fontsize=20)
plt.xlabel("Time", fontsize=18)
plt.ylabel("Data", fontsize=18)
plt.show()


# From the above graph (signal in red and open channels in blue), we can see that there is a clear link between the volatility in signal and the volatility in open channels. Above *t = 300*, we can see a trend of the form *y = sin<sup>2</sup>(x)*, but with monotonously increasing amplitudes in the signal data. The period of the signal's oscillation perfectly matches with that of the open channels. The maximum value of open channels tends to increase as the amplitude of the signal increasing, giving the *open_channels* graph a "step function" sort of shape above *t = 300*. This is clear evidence of the effect of signal volatility on the ion channels.

# ## Measures of complexity (entropy and fractal dimension) <a id="2.5"></a>
# 
# Entropy and fractal dimension are methods of measuring the "roughness" or "complexity" of a signal. I will now explore the relationships of these features with the number of open channels.

# ### Permutation entropy
# 
# The permutation entropy is a complexity measure for time-series first introduced by Bandt and Pompe in 2002. It represents the information contained in comparing n consecutive values of the time series. It is a measure of entropy or disorderliness in a time series.

# In[ ]:


def _embed(x, order=3, delay=1):
    N = len(x)
    if order * delay > N:
        raise ValueError("Error: order * delay should be lower than x.size")
    if delay < 1:
        raise ValueError("Delay has to be at least 1.")
    if order < 2:
        raise ValueError("Order has to be at least 2.")
    Y = np.zeros((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T

all = ['perm_entropy', 'spectral_entropy', 'svd_entropy', 'app_entropy',
       'sample_entropy']


def perm_entropy(x, order=3, delay=1, normalize=False):
    x = np.array(x)
    ran_order = range(order)
    hashmult = np.power(order, ran_order)
    # Embed x and sort the order of permutations
    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind='quicksort')
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    # Return the counts
    _, c = np.unique(hashval, return_counts=True)
    # Use np.true_divide for Python 2 compatibility
    p = np.true_divide(c, c.sum())
    pe = -np.multiply(p, np.log2(p)).sum()
    if normalize:
        pe /= np.log2(factorial(order))
    return pe


# ### Permutation entropy vs. Open channels

# In[ ]:


df = pd.DataFrame(np.transpose([[perm_entropy(row) for row in signals], targets]))
df.columns = ["perm_entropy", "open_channels"]
fig = go.Figure()

channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for channel in channels:
    fig.add_trace(go.Box(x=df['open_channels'][df['open_channels'] == channel],
                         y=df['perm_entropy'][df['open_channels'] == channel],
                         name=channel,
                         marker=dict(color='blueviolet'), showlegend=False)
                         )
    
fig.add_trace(go.Scatter(x=channels,
                         y=[df['perm_entropy'][df['open_channels'] == channel].median() for channel in channels],
                         mode="lines+markers",
                         name=channel,
                         marker=dict(color='blueviolet'), showlegend=False)
                         )

fig.update_layout(title="Permutation entropy vs. Open channels", xaxis_title="Open channels", yaxis_title="Permutation entropy")
fig.show()


# In[ ]:


df = pd.DataFrame(np.transpose([[perm_entropy(row) for row in signals], targets]))
df.columns = ["perm_entropy", "open_channels"]
fig = go.Figure()

channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

fig.add_trace(go.Scatter(x=channels,
                         y=[df['perm_entropy'][df['open_channels'] == channel].median() for channel in channels],
                         mode="lines+markers",
                         name=channel,
                         marker=dict(color='blueviolet'), showlegend=False)
                         )

fig.update_layout(title="Median permutation entropy vs. Open channels", xaxis_title="Open channels", yaxis_title="Median permutation entropy")
fig.show()


# From the above graphs, we can see that the distribution of permutation entropy is different for different values of *open_channels*. The distribution at *open_channels = 10* has the lowest median. This suggests that higher *open_channels* values are linked with less complex or volatile signals. The only exception to this trend is *open_channels = 6* where we see a sharp dip in permutation entropy. Besides this, all distributions seem to have a median between 2.56 and 2.58. All distributions have a clear leftward (negative) skew as well.

# ### Approximate entropy
# 
# Approximate entropy is a technique used to quantify the amount of regularity and the unpredictability of fluctuations over time-series data. Smaller values indicates that the data is more regular and predictable.

# In[ ]:


def _app_samp_entropy(x, order, metric='chebyshev', approximate=True):
    """Utility function for `app_entropy`` and `sample_entropy`.
    """
    _all_metrics = KDTree.valid_metrics
    if metric not in _all_metrics:
        raise ValueError('The given metric (%s) is not valid. The valid '
                         'metric names are: %s' % (metric, _all_metrics))
    phi = np.zeros(2)
    r = 0.2 * np.std(x, axis=-1, ddof=1)

    # compute phi(order, r)
    _emb_data1 = _embed(x, order, 1)
    if approximate:
        emb_data1 = _emb_data1
    else:
        emb_data1 = _emb_data1[:-1]
    count1 = KDTree(emb_data1, metric=metric).query_radius(emb_data1, r,
                                                           count_only=True
                                                           ).astype(np.float64)
    # compute phi(order + 1, r)
    emb_data2 = _embed(x, order + 1, 1)
    count2 = KDTree(emb_data2, metric=metric).query_radius(emb_data2, r,
                                                           count_only=True
                                                           ).astype(np.float64)
    if approximate:
        phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
        phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
    return phi


def _numba_sampen(x, mm=2, r=0.2):
    """
    Fast evaluation of the sample entropy using Numba.
    """
    n = x.size
    n1 = n - 1
    mm += 1
    mm_dbld = 2 * mm

    # Define threshold
    r *= x.std()

    # initialize the lists
    run = [0] * n
    run1 = run[:]
    r1 = [0] * (n * mm_dbld)
    a = [0] * mm
    b = a[:]
    p = a[:]

    for i in range(n1):
        nj = n1 - i

        for jj in range(nj):
            j = jj + i + 1
            if abs(x[j] - x[i]) < r:
                run[jj] = run1[jj] + 1
                m1 = mm if mm < run[jj] else run[jj]
                for m in range(m1):
                    a[m] += 1
                    if j < n1:
                        b[m] += 1
            else:
                run[jj] = 0
        for j in range(mm_dbld):
            run1[j] = run[j]
            r1[i + n * j] = run[j]
        if nj > mm_dbld - 1:
            for j in range(mm_dbld, nj):
                run1[j] = run[j]

    m = mm - 1

    while m > 0:
        b[m] = b[m - 1]
        m -= 1

    b[0] = n * n1 / 2
    a = np.array([float(aa) for aa in a])
    b = np.array([float(bb) for bb in b])
    p = np.true_divide(a, b)
    return -log(p[-1])


def app_entropy(x, order=2, metric='chebyshev'):
    phi = _app_samp_entropy(x, order=order, metric=metric, approximate=True)
    return np.subtract(phi[0], phi[1])


# ### Approximate entropy vs. Open channels

# In[ ]:


df = pd.DataFrame(np.transpose([[app_entropy(row) for row in signals], targets]))
df.columns = ["app_entropy", "open_channels"]
fig = go.Figure()

channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for channel in channels:
    fig.add_trace(go.Box(x=df['open_channels'][df['open_channels'] == channel],
                         y=df['app_entropy'][df['open_channels'] == channel],
                         name=channel,
                         marker=dict(color='tomato'), showlegend=False)
                         )
    
fig.add_trace(go.Scatter(x=channels,
                         y=[df['app_entropy'][df['open_channels'] == channel].median() for channel in channels],
                         mode="lines+markers",
                         name=channel,
                         marker=dict(color='tomato'), showlegend=False)
                         )

fig.update_layout(title="Approximate entropy vs. Open channels", xaxis_title="Open channels", yaxis_title="Approximate entropy")
fig.show()


# In[ ]:


df = pd.DataFrame(np.transpose([[app_entropy(row) for row in signals], targets]))
df.columns = ["app_entropy", "open_channels"]
fig = go.Figure()

channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

fig.add_trace(go.Scatter(x=channels,
                         y=[df['app_entropy'][df['open_channels'] == channel].median() for channel in channels],
                         mode="lines+markers",
                         name=channel,
                         marker=dict(color='tomato'), showlegend=False)
                         )

fig.update_layout(title="Median approximate entropy vs. Open channels", xaxis_title="Open channels", yaxis_title="Median approximate entropy")
fig.show()


# From the above graphs, we can see that the distribution of approximate entropy is different for different values of *open_channels*. Approximate entropy, however, does not seem to have a clear relationship with *open_channels*. Besides this, all distributions seem to have a median between 1.45 and 1.65. All distributions are roughly normal as well.

# ### Higuchi fractal dimension
# 
# The Higuchi fractal dimension is a method to calculate the fractal dimension of any two-dimensional curve. Generally, curves with higher fractal dimension are "rougher" or more "complex" (higher entropy).

# In[ ]:


def _log_n(min_n, max_n, factor):
    max_i = int(floor(log(1.0 * max_n / min_n) / log(factor)))
    ns = [min_n]
    for i in range(max_i + 1):
        n = int(floor(min_n * (factor ** i)))
        if n > ns[-1]:
            ns.append(n)
    return np.array(ns, dtype=np.int64)

def _higuchi_fd(x, kmax):
    n_times = x.size
    lk = np.empty(kmax)
    x_reg = np.empty(kmax)
    y_reg = np.empty(kmax)
    for k in range(1, kmax + 1):
        lm = np.empty((k,))
        for m in range(k):
            ll = 0
            n_max = floor((n_times - m - 1) / k)
            n_max = int(n_max)
            for j in range(1, n_max):
                ll += abs(x[m + j * k] - x[m + (j - 1) * k])
            ll /= k
            ll *= (n_times - 1) / (k * n_max)
            lm[m] = ll
        # Mean of lm
        m_lm = 0
        for m in range(k):
            m_lm += lm[m]
        m_lm /= k
        lk[k - 1] = m_lm
        x_reg[k - 1] = log(1. / k)
        y_reg[k - 1] = log(m_lm)
    higuchi, _ = _linear_regression(x_reg, y_reg)
    return higuchi


def higuchi_fd(x, kmax=10):
    x = np.asarray(x, dtype=np.float64)
    kmax = int(kmax)
    return _higuchi_fd(x, kmax)

def _linear_regression(x, y):
    n_times = x.size
    sx2 = 0
    sx = 0
    sy = 0
    sxy = 0
    for j in range(n_times):
        sx2 += x[j] ** 2
        sx += x[j]
        sxy += x[j] * y[j]
        sy += y[j]
    den = n_times * sx2 - (sx ** 2)
    num = n_times * sxy - sx * sy
    slope = num / den
    intercept = np.mean(y) - slope * np.mean(x)
    return slope, intercept


# ### Higuchi fractal dimension vs. Open channels

# In[ ]:


df = pd.DataFrame(np.transpose([[higuchi_fd(row) for row in signals], targets]))
df.columns = ["higuchi_fd", "open_channels"]
fig = go.Figure()

channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for channel in channels:
    fig.add_trace(go.Box(x=df['open_channels'][df['open_channels'] == channel],
                         y=df['higuchi_fd'][df['open_channels'] == channel],
                         name=channel,
                         marker=dict(color='orange'), showlegend=False)
                         )
    
fig.add_trace(go.Scatter(x=channels,
                         y=[df['higuchi_fd'][df['open_channels'] == channel].median() for channel in channels],
                         mode="lines+markers",
                         name=channel,
                         marker=dict(color='orange'), showlegend=False)
                         )

fig.update_layout(title="Higuchi fractal dimension vs. Open channels", xaxis_title="Open channels", yaxis_title="Higuchi fractal dimension")
fig.show()


# In[ ]:


df = pd.DataFrame(np.transpose([[higuchi_fd(row) for row in signals], targets]))
df.columns = ["higuchi_fd", "open_channels"]
fig = go.Figure()

channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

fig.add_trace(go.Scatter(x=channels,
                         y=[df['higuchi_fd'][df['open_channels'] == channel].median() for channel in channels],
                         mode="lines+markers",
                         name=channel,
                         marker=dict(color='orange'), showlegend=False)
                         )

fig.update_layout(title="Median Higuchi fractal dimension vs. Open channels", xaxis_title="Open channels", yaxis_title="Median Higuchi fractal dimension")
fig.show()


# From the above graphs, we can see that *open_channels* decreases with increase in Higuchi fractal dimension. Besides this, most distributions have a median of approximately 2 and roughly symmetric bell shape (normal).

# ### Katz fractal dimension
# 
# The Katz fractal dimension is yet another way to calculate the fractal dimension of a two-dimensional curve.

# In[ ]:


def katz_fd(x):
    x = np.array(x)
    dists = np.abs(np.ediff1d(x))
    ll = dists.sum()
    ln = np.log10(np.divide(ll, dists.mean()))
    aux_d = x - x[0]
    d = np.max(np.abs(aux_d[1:]))
    return np.divide(ln, np.add(ln, np.log10(np.divide(d, ll))))


# ### Katz fractal dimension vs. Open channels

# In[ ]:


df = pd.DataFrame(np.transpose([[katz_fd(row) for row in signals], targets]))
df.columns = ["katz_fd", "open_channels"]
fig = go.Figure()

channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for channel in channels:
    fig.add_trace(go.Box(x=df['open_channels'][df['open_channels'] == channel],
                         y=df['katz_fd'][df['open_channels'] == channel],
                         name=channel,
                         marker=dict(color='teal'), showlegend=False)
                         )
    
fig.add_trace(go.Scatter(x=channels,
                         y=[df['katz_fd'][df['open_channels'] == channel].median() for channel in channels],
                         mode="lines+markers",
                         name=channel,
                         marker=dict(color='teal'), showlegend=False)
                         )

fig.update_layout(title="Katz fractal dimension vs. Open channels", xaxis_title="Open channels", yaxis_title="Katz fractal dimension")
fig.show()


# In[ ]:


df = pd.DataFrame(np.transpose([[katz_fd(row) for row in signals], targets]))
df.columns = ["katz_fd", "open_channels"]
fig = go.Figure()

channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

fig.add_trace(go.Scatter(x=channels,
                         y=[df['katz_fd'][df['open_channels'] == channel].median() for channel in channels],
                         mode="lines+markers",
                         name=channel,
                         marker=dict(color='teal'), showlegend=False)
                         )

fig.update_layout(title="Median Katz fractal dimension vs. Open channels", xaxis_title="Open channels", yaxis_title="Median Katz fractal dimension")
fig.show()


# From the above graphs, we can see that the Katz fractal dimension also decreases with increase in the *open_channels* variable (similar to Higuchi FD). Besides this, most distributions have a rightward (positive) skew and a median between 3.8 and 5.

# # Ending note <a id="3"></a>
# 
# <font size=4 color="red">This concludes my EDA kernel. Please upvote this kernel if you like it. It motivates me to produce more quality content :)</font>
