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


# In[ ]:


df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')


# # 1. Preprocessing

# In[ ]:


df.info()


# In[ ]:


df.head()


# #### We can see some data of object type, and especially column 'Id, 'groupId', and 'matchId' don't seem any significance for prediction. So just exclude those columns.

# In[ ]:


df = df.drop(df.columns[[0,1,2]],axis=1)


# #### Also, there is still an object type column - 'matchType'. Remember to transform it to integer type by LabelEncoder before modeling!!

# In[ ]:


# from sklearn.preprocessing import LabelEncoder
# le=LabelEncoder()
# df['matchType']=le.fit_transform(df['matchType'])


# In[ ]:


df.info()


# In[ ]:


df.describe()


# # 2. EDA

# In[ ]:


import matplotlib.pyplot as plt       
import seaborn as sns                
import warnings                       
warnings.filterwarnings("ignore")


# #### Pandans profiling library is really helpful for you if you wanna check out correlation between features, missing values, etc... at once.

# In[ ]:


# import pandas_profiling as pp
# pp.ProfileReport(df)


# ### Data fields
# * DBNOs - Number of enemy players knocked.
# * assists - Number of enemy players this player damaged that were killed by teammates.
# * boosts - Number of boost items used.
# * damageDealt - Total damage dealt. Note: Self inflicted damage is subtracted.
# * headshotKills - Number of enemy players killed with headshots.
# * heals - Number of healing items used.
# * killPlace - Ranking in match of number of enemy players killed.
# * killPoints - Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.) If there is a value other than -1 in rankPoints, then any 0 in killPoints should be treated as a “None”.
# * killStreaks - Max number of enemy players killed in a short amount of time.
# * kills - Number of enemy players killed.
# * longestKill - Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.
# * matchType - String identifying the game mode that the data comes from. The standard modes are “solo”, “duo”, “squad”, “solo-fpp”, duo-fpp”, and “squad-fpp”; other modes are from events or custom matches.
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
# * numGroups - Number of groups we have data for in the match.
# * maxPlace - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.
# * winPlacePerc - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.

# In[ ]:


df = df.drop('matchDuration',axis=1)


# In[ ]:


df['matchType'].value_counts()


# ### What is FPP or TPP?
# 
# Unlike most other hardcore first-person shooter (FPS) games, which only have first-person view, PUBG Mobile gives players an option to change to third-person camera. Both Third Person Perspective (TPP) and First Person Perspective (FPP) have different matches and separate ranking systems.

# In[ ]:


mt_list = pd.Series.tolist(df['matchType'])
new_list = []
for v in mt_list:
    if v not in new_list:
        new_list.append(v)
print(new_list)


# In[ ]:


plt.figure(figsize=(15,8))
ax = sns.countplot(x='matchType', data=df, order = df['matchType'].value_counts().index)
ax.set_xticklabels(new_list,rotation=60, fontsize=14)
ax.set_title("Match types")
plt.show()


# #### Flaretpp and crashtpp are supposed to event mode, normal-solo similiar solo.
# 
# * flare (flare gun) : Flare Gun can be obtained. A flare gun is a kind of signal that shoots toward the sky. It is a special firearm that supplies a special package (Care Package) when the airplane is deployed to that position when in use.
# 
# * crash (Crash Carnage) : " Road scary is the subject of this week's event mode, where fuel and fire are king." In Crashcanage, there are no firearms, so melee weapons, drones, You have to concentrate on driving skills that can move the Duo to the final round. In this event, the circle moves quite fast, so it quickly crashes on the way to the loot, the vehicle, and the warrior glory of the road. " i.e. It means to win only with a melee weapon without a gun.
# 
# https://www.kaggle.com/c/pubg-finish-placement-prediction/discussion/69700

# In[ ]:


types = df["matchType"].value_counts().to_frame()
squads = types.loc[["squad-fpp","squad","normal-squad-fpp","normal-squad"],"matchType"].sum()
duos = types.loc[["duo-fpp","duo","normal-duo-fpp","normal-duo"],"matchType"].sum()
solo = types.loc[["solo-fpp","solo","normal-solo-fpp","normal-solo"],"matchType"].sum()
mt = pd.DataFrame([squads, duos, solo], index=["squad","duo","solo"], columns =["count"])
mt


# In[ ]:


## 데이터 준비
labels = ['squad', 'duo', 'solo'] ## 라벨
frequency = pd.Series.tolist(mt['count'])

fig = plt.figure(figsize=(8,8)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 배경색을 하얀색으로 설정
ax = fig.add_subplot() ## 프레임 생성
 
ax.pie(frequency, ## 파이차트 출력
       labels=labels, ## 라벨 출력
       startangle=90, ## 시작점을 90도(degree)로 지정
       counterclock=False, ## 시계 방향으로 그린다.
       autopct=lambda p : '{:.2f}%'.format(p) ## 퍼센티지 출력
       )
 
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
ax = sns.distplot(df["numGroups"])
ax.set_title("Number of groups")
plt.show()


# ### Correlation

# In[ ]:


f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# #### As you know, negative correlation is meaningless... So just focus on some features with positive correlation.

# In[ ]:


k = 5 #number of variables for heatmap
cols = df.corr().nlargest(k, 'winPlacePerc')['winPlacePerc'].index
cols


# In[ ]:


new_df = df[['winPlacePerc', 'walkDistance', 'boosts', 'weaponsAcquired',
       'damageDealt']]


# In[ ]:


new_df


# In[ ]:


corr = new_df.corr()
# 그림 사이즈 지정
fig, ax = plt.subplots( figsize=(7,7) )

# 삼각형 마스크를 만든다(위 쪽 삼각형에 True, 아래 삼각형에 False)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# 히트맵을 그린다
sns.heatmap(corr, 
            cmap = 'RdYlBu_r', 
            annot = True,   # 실제 값을 표시한다
            mask=mask,      # 표시하지 않을 마스크 부분을 지정한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
            vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
           )  
plt.show()


# In[ ]:


sns.set()
cols = ['winPlacePerc', 'walkDistance', 'boosts', 'weaponsAcquired', 'damageDealt', 'killPlace']
sns.pairplot(df[cols], size = 2.5)
plt.show()


# In[ ]:




