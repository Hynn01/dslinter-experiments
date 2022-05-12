#!/usr/bin/env python
# coding: utf-8

# # Cox proportional hazard model

# The yardage gained on the play distributes to a one-dimenstional distribution which depends on rushing plays information. This notebook adopts **Cox proportional hazard model** to express this distribution and shows the advantage of suvival analysis approach in this competition.

# Survival analysis mainly focus on explaining the duration of time until some events happen. In survival analysis, our aim is to express the hazard function instead of probability density function, which is defined as
# 
# $$
# \begin{aligned}
# h(t) = \lim_{\Delta t \to 0} \frac{\mathrm{Pr}(t \leq T < t + \Delta t \mid T \geq t)}{\Delta t}, \quad t \geq 0.
# \end{aligned}
# $$
# 
# **Cox proportional model** divides this hazard function into two parts as
# 
# $$
# \begin{aligned}
#     &h(t) = h_0(t) \exp(X \beta), \quad \\
#     &\text{$X$ : covariates, $\beta$: parameter.}
# \end{aligned}
# $$
# 
# 
# The former part indicates how the hazard function depends on time and the latter part indicates how it depends on the covariate information.  Estimating the former part in empirical manner enables us to express the complex distribution easily.

# In this notebook, we consider that the distance obtained by ball carrier as duration of time until event happens. We adopt simple linear model with players location, velocity, acceralation, counts of down, team as covariates.

# # Setup

# In[ ]:


import numpy as np
import pandas as pd
import os
import statsmodels.api as sm

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Define function for etracting features from raw dataset 

# In[ ]:


def extract_feature(play, is_train=True):
    
    if play['PlayDirection'].iloc[0] == 'right':
        direction = 1
    else:
        direction = -1
        
    home, away = play['Team'].values == 'home', play['Team'].values == 'away'
    indRusher = np.where(play['NflId'].values == play['NflIdRusher'].iloc[0])[0][0]

    if play['FieldPosition'].iloc[0] == play['PossessionTeam'].iloc[0]:
        yardToGoal = 100 - play['YardLine'].iloc[0]
        start = np.array([120 + (play['YardLine'].iloc[0] + 10) * direction, 53.3 / 2]) % 120
    else:
        yardToGoal = play['YardLine'].iloc[0]
        start = np.array([120 - (play['YardLine'].iloc[0] + 10) * direction, 53.3 / 2]) % 120

    Dir = play['Dir'].values
    rad = np.nan_to_num(2 * np.pi * (90 - Dir) / 360)
    x, y = play['X'].values, play['Y'].values
    S = play['S'].values * np.logical_not(np.isnan(Dir))
    A = play['A'].values * np.logical_not(np.isnan(Dir))

    loc = np.vstack([x - start[0], y - start[1]]).T * direction
    vel = (S * np.vstack([np.cos(rad), np.sin(rad)])).T * direction
    acc = (A * np.vstack([np.cos(rad), np.sin(rad)])).T * direction
    locRusher, velRusher, accRusher = loc[indRusher], vel[indRusher], acc[indRusher]

    diff = np.hstack([np.square(loc - locRusher), np.square(vel - velRusher), np.square(acc - accRusher)])

    scrimWidth = 5
    inTheBox = (play['NflId'].values != play['NflIdRusher'].iloc[0]) * (np.abs(loc[:, 0]) < scrimWidth)

    locDet = np.linalg.slogdet(np.exp(- np.square(loc[inTheBox, np.newaxis] - loc[inTheBox][np.newaxis]).sum(2) / 2.))[1]
    locHomeDet = np.linalg.slogdet(np.exp(- np.square(loc[home * inTheBox][:, np.newaxis] - loc[home * inTheBox][np.newaxis]).sum(2) / 2.))[1]
    locAwayDet = np.linalg.slogdet(np.exp(- np.square(loc[away * inTheBox][:, np.newaxis] - loc[away * inTheBox][np.newaxis]).sum(2) / 2.))[1]

    if play['PossessionTeam'].iloc[0] == play['HomeTeamAbbr'].iloc[0]:
        x = np.hstack([diff[home].sum(0), diff[away].sum(0), locDet, locHomeDet, locAwayDet])
    else: 
        x = np.hstack([diff[away].sum(0), diff[home].sum(0), locDet, locAwayDet, locHomeDet])
        
    x = np.hstack([locRusher, velRusher, accRusher, x, (downs == play['Down'].iloc[0]).astype(np.float), (teams == play['PossessionTeam'].iloc[0]).astype(np.float)])    
    
    offset = locRusher[0] - 5
    threshold = play['Distance'].iloc[0] - offset
    
    if is_train:
         
        yard = play['Yards'].iloc[0] - offset
        
        c = yard < threshold
        y = np.minimum(yard, threshold)
        
        return x, y, c, offset
    
    else:
        return x, offset


# # Load data

# In[ ]:


data = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

data.loc[data.HomeTeamAbbr.values == "ARI", 'HomeTeamAbbr'] = "ARZ"
data.loc[data.HomeTeamAbbr.values == "BAL", 'HomeTeamAbbr'] = "BLT"
data.loc[data.HomeTeamAbbr.values == "CLE", 'HomeTeamAbbr'] = "CLV"
data.loc[data.HomeTeamAbbr.values == "HOU", 'HomeTeamAbbr'] = "HST"

data.loc[data['Season'] == 2017, 'S'] = (data['S'][data['Season'] == 2017] - 2.4355) / 1.2930 * 1.4551 + 2.7570

downs = np.array([1, 2, 3])
teams = pd.get_dummies(data['PossessionTeam']).columns[:-1]

train = data
n_train = train.shape[0] // 22


# # Extract features from training dataset

# In[ ]:


inds = list(train.groupby('PlayId').groups.values())

xs, ys, cs = [], [], []

for i in range(n_train):

    ind = inds[i]
    play = train.loc[ind]
    x, y, c, _ = extract_feature(play)

    xs.append(x)
    ys.append(y)
    cs.append(c)

xs, ys, cs = np.vstack(xs), np.hstack(ys), np.array(cs).astype(np.int)
ys = np.maximum(0, ys)


# # Estimate parameters of Cox proportional model

# In[ ]:


model = sm.PHReg(ys, xs, cs)
result = model.fit()

baseline_cum_hazard_func = result.baseline_cumulative_hazard_function[0]
pred_index = np.arange(-99, 100)


# # Predict 

# In[ ]:


from kaggle.competitions import nflrush
env = nflrush.make_env()


# In[ ]:


for (play, prediction_df) in env.iter_test():
    
    play.loc[play.HomeTeamAbbr.values == "ARI", 'HomeTeamAbbr'] = "ARZ"
    play.loc[play.HomeTeamAbbr.values == "BAL", 'HomeTeamAbbr'] = "BLT"
    play.loc[play.HomeTeamAbbr.values == "CLE", 'HomeTeamAbbr'] = "CLV"
    play.loc[play.HomeTeamAbbr.values == "HOU", 'HomeTeamAbbr'] = "HST"
    
    x, offset = extract_feature(play, False)
    
    cum_hazard = np.exp(result.params.dot(x)) * baseline_cum_hazard_func(pred_index - offset)
    pred = 1 - np.exp(- cum_hazard)
    
    if play['FieldPosition'].iloc[0] == play['PossessionTeam'].iloc[0]:
        yardToGoal = 100 - play['YardLine'].iloc[0]
    else:
        yardToGoal = play['YardLine'].iloc[0]

    pred /= pred[pred_index <= yardToGoal][-1]
    pred[pred_index > yardToGoal] = 1.
    
    prediction_df = pd.DataFrame(pred[np.newaxis], columns=prediction_df.columns)
    
    env.predict(prediction_df)


# In[ ]:


env.write_submission_file()

