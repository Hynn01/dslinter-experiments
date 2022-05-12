#!/usr/bin/env python
# coding: utf-8

# # 배틀그라운드 EDA, 시각화
# 

# In[ ]:


# Data and arrays handling
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Interactive plotting
from plotly.offline import init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objs as go

init_notebook_mode(connected=True)

# Inline plots
get_ipython().run_line_magic('matplotlib', 'inline')

# Ignore warns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')


# In[ ]:


train.info()


# ## Data fields
# * DBNOs - Number of enemy players knocked.
# * assists - Number of enemy players this player damaged that were killed by teammates.
# * boosts - Number of boost items used.
# * damageDealt - Total damage dealt. Note: Self inflicted damage is subtracted.
# * headshotKills - Number of enemy players killed with headshots.
# * heals - Number of healing items used.
# * Id - Player’s Id
# * killPlace - Ranking in match of number of enemy players killed.
# * killPoints - Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.) If there is a value other than -1 in rankPoints, then any 0 in killPoints should be treated as a “None”.
# * killStreaks - Max number of enemy players killed in a short amount of time.
# * kills - Number of enemy players killed.
# * longestKill - Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.
# * matchDuration - Duration of match in seconds.
# * matchId - ID to identify match. There are no matches that are in both the training and testing set.
# * matchType - String identifying the game mode that the data comes from. The standard modes are “solo”, “duo”, “squad”, “solo-fpp”, “duo-fpp”, and “squad-fpp”; other modes are from events or custom matches.
# * rankPoints - Elo-like ranking of player. This ranking is inconsistent and is being deprecated in the API’s next version, so use with caution. Value of -1 takes place of “None”.
# * revives - Number of times this player revived teammates.
# * rideDistance - Total distance traveled in vehicles measured in meters.
# * roadKills - Number of kills while in a vehicle.
# * swimDistance - Total distance traveled by swimming measured in meters.
# * teamKills - Number of times this player killed a teammate.
# * vehicleDestroys - Number of vehicles destroyed.
# * walkDistance - Total distance traveled on foot measured in meters.
# * weaponsAcquired - Number of weapons picked up.
# * winPoints - Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.) If there is a value other than -1 in rankPoints, then any 0 in winPoints should be treated as a “None”.
# * groupId - ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.
# * numGroups - Number of groups we have data for in the match.
# * maxPlace - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.
# * winPlacePerc - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.

# In[ ]:


train.head()


# In[ ]:


# 결측치 확인하기
train.isna().sum()


# # 킬

# In[ ]:


# Kill 수

data = train.copy() # .copy() 사본 생성
# loc = 행 or 열 조회하기
data.loc[data['kills'] > data['kills'].quantile(0.99)] = '8+' 
# 99% 이상의 값들은 8+로 계산

plt.figure(figsize=(15,10))
sns.countplot(data['kills'].astype('str').sort_values())
plt.title("Kill Count",fontsize=15)
plt.show()


# In[ ]:


# 0킬인 사람들은 피해를 얼마나 입을까?
data = train.copy()
data = data[data['kills']==0]
plt.figure(figsize=(15,10))
plt.title("Damage Dealt by 0 killers",fontsize=15)
sns.distplot(data['damageDealt'])
plt.show()


# In[ ]:


print("{} players ({:.4f}%) have won without a single kill!".format(len(data[data['winPlacePerc']==1]), 100*len(data[data['winPlacePerc']==1])/len(train)))

data1 = train[train['damageDealt'] == 0].copy()
print("{} players ({:.4f}%) have won without dealing damage!".format(len(data1[data1['winPlacePerc']==1]), 100*len(data1[data1['winPlacePerc']==1])/len(train)))


# In[ ]:


sns.jointplot(x="winPlacePerc", y="kills", data=train, height=10, ratio=3, color="r")
plt.show()


# In[ ]:


# 0킬, 1-2킬, 3-5킬, 6-10킬, 10킬 이상 분포 확인해보기
kills = train.copy()

# pd.cut() = 동일한 길이로 나누기
# pd.cut(데이터, 구간의 갯수, 레이블명)
kills['killsCategories'] = pd.cut(kills['kills'], [-1, 0, 2, 5, 10, 60], labels=['0_kills','1-2_kills', '3-5_kills', '6-10_kills', '10+_kills'])

plt.figure(figsize=(15,8))
sns.boxplot(x="killsCategories", y="winPlacePerc", data=kills)
plt.show()


# In[ ]:


def featStat(featureName, constrain,plotType):
    feat = train[featureName][train[featureName]>0]
    data = train[[featureName,'winPlacePerc']].copy()
    q99 = int(data[featureName].quantile(0.99))
    plt.rcParams['figure.figsize'] = 15,5;   
    
    if constrain!=None:
        feat = feat[feat<constrain]
    if plotType == 'hist':
        plt.subplot(1,2,1)
        feat.hist(bins=50);
        plt.title(featureName);
        
        n = 20
        cut_range = np.linspace(0,q99,n)
        cut_range = np.append(cut_range, data[featureName].max())
        data[featureName] = pd.cut(data[featureName],
                                         cut_range,
                                         labels=["{:.0f}-{:.0f}".format(a_, b_) for a_, b_ in zip(cut_range[:n], cut_range[1:])],
                                         include_lowest=True
                                        )
        ax = plt.subplot(1,2,2)
        sns.boxplot(x="winPlacePerc", y=featureName, data=data, ax=ax, color="#2196F3")
        ax.set_xlabel('winPlacePerc', size=14, color="#263238")
        ax.set_ylabel(featureName, size=14, color="#263238")
        plt.gca().xaxis.grid(True)
        plt.tight_layout()
           
    if plotType == 'count':        
        plt.subplot(1,2,1)
        sns.countplot(feat, color="#2196F3");
        
        plt.subplot(1,2,2)
        data.loc[data[featureName] > q99, featureName] = q99+1
        x_order = data.groupby(featureName).mean().reset_index()[featureName]
        x_order.iloc[-1] = str(q99+1)+"+"
        data[featureName][data[featureName] == q99+1] = str(q99+1)+"+"
        
        ax = sns.boxplot(x=featureName, y='winPlacePerc', data=data, color="#2196F3", order = x_order);
        ax.set_xlabel(featureName, size=14, color="#263238")
        ax.set_ylabel('WinPlacePerc', size=14, color="#263238")
    plt.tight_layout()


# In[ ]:


featStat('kills',15,'count');
plt.show();
featStat('longestKill',400,'hist');
plt.show();
featStat('damageDealt',1000,'hist');


# ## 달리기

# In[ ]:


print("The average person walks for {:.1f}m, 99% of people have walked {}m or less, while the marathoner champion walked for {}m.".format(train['walkDistance'].mean(), train['walkDistance'].quantile(0.99), train['walkDistance'].max()))


# In[ ]:


data = train.copy()
data = data[data['walkDistance'] < train['walkDistance'].quantile(0.99)]
plt.figure(figsize=(15,10))
plt.title("Walking Distance Distribution",fontsize=15)
sns.distplot(data['walkDistance'])
plt.show()


# In[ ]:


print("{} players ({:.4f}%) walked 0 meters. This means that they die before even taking a step or they are afk (more possible).".format(len(data[data['walkDistance'] == 0]), 100*len(data1[data1['walkDistance']==0])/len(train)))


# In[ ]:


sns.jointplot(x="winPlacePerc", y="walkDistance",  data=train, height=10, ratio=3, color="lime")
plt.show()


# In[ ]:


featStat('walkDistance',5000,'hist')
plt.show()
featStat('swimDistance',500,'hist')
plt.show()
featStat('rideDistance',12000,'hist')


# # 회복과 부스트

# In[ ]:


featStat('heals',20,'count')
plt.show()
featStat('boosts',12,'count')

