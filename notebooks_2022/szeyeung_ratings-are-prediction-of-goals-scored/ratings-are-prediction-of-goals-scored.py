#!/usr/bin/env python
# coding: utf-8

# # Are the Ratings Predictions of Goals Scored?

# In this competition of predicting football match results primarily based on the data of past ten matches of the two sides, the rating feature remains a mystery for many participants. Octosport, the host of this competition, states in the description of the dataset that:
# 
# > The rating features are calculated by Octosport. Ratings are meant to give information on the team's relative strength for a given match. For instance, we would expect that a team with a rating of 10.5 beats a team with a rating of 2.3.
# 
# In a [discussion](https://www.kaggle.com/code/curiosityquotient/investigation-into-rating-feature-octosport/comments), it is further revealed that:
# 
# > ...ratings are pairwise and can be seen as a combination of a team's offensive strength and an opponent's defensive strength which is why they are varying in time depending on the fixture.
# 
# A research of football prediction models shows that this kind of formulation may be more likely associated with the Poisson models, which use the assessments of the offensive and defensive strength to come up with the expected mean goals of each side, then use Poisson distribution to turn out the probabilities of every possible score lines and thereby the probabilities of home win, draw or away win. Do the rating features here operate in this way? Let's see what the data tell us.

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


#import modules
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

get_ipython().run_line_magic('matplotlib', 'inline')

#loading the data
train = pd.read_csv('/kaggle/input/football-match-probability-prediction/train.csv')
test = pd.read_csv('/kaggle/input/football-match-probability-prediction/test.csv')


# In this analysis I only investigate the train set. In the original dataset, the ratings of past ten matches of the home team and its opponents and the away team and its opponents are listed in 40 columns in a 'wide' form. In this investigation of the relationship between ratings and final score lines, it may be more convenient to 'stack' them into a 'long' form.

# In[ ]:


history_columns=['id']+[col for col in train.columns if col[-1].isdigit()]


# In[ ]:


stubname=['home_team_history_match_date',
           'home_team_history_is_play_home',
           'home_team_history_is_cup',
           'home_team_history_goal',
           'home_team_history_opponent_goal',
           'home_team_history_rating',
           'home_team_history_opponent_rating',
           'home_team_history_coach',
           'home_team_history_league_id',
           'away_team_history_match_date',
           'away_team_history_is_play_home',
           'away_team_history_is_cup',
           'away_team_history_goal',
           'away_team_history_opponent_goal',
           'away_team_history_rating',
           'away_team_history_opponent_rating',
           'away_team_history_coach',
           'away_team_history_league_id']

history_train=pd.wide_to_long(train[history_columns], stubnames=stubname, i='id', j='match', sep='_')
history_train=history_train.reset_index().sort_values(by=['id','match'])
history_train


# In[ ]:


history_train.duplicated(subset=stubname).value_counts()


# I originally thought that there might be many duplicated records, but the counts do not go this way. So I leave them as they are.

# In[ ]:


rating_columns=[col for col in history_train if 'rating' in col]
history_train[rating_columns].describe()


# The distributions of ratings are very skewed, ranging from 0.21 to more than 57, with mean about 6.5 and medium about 6.

# In[ ]:


# adding features for rating gaps and goal differences
history_train['home_rating_gap']= history_train['home_team_history_rating']-history_train['home_team_history_opponent_rating']
history_train['home_goal_difference']= history_train['home_team_history_goal']-history_train['home_team_history_opponent_goal']
history_train['away_rating_gap']= history_train['away_team_history_rating']-history_train['away_team_history_opponent_rating']
history_train['away_goal_difference']= history_train['away_team_history_goal']-history_train['away_team_history_opponent_goal']


# In[ ]:


sns.scatterplot(data=history_train, x='home_rating_gap', y='home_goal_difference')


# A scatter plot on rating gaps and also the final goal differences between the home teams and their past opponents generally form a cloud of tilted ellipse. There is a general trend that a large rating gap results in a large goal difference, but there are also a wide range of possibilities, the teams with equal rating can end up in draws, a win or a lose up to 10 goals in margin. A similar pattern exists for the away teams and their opponents.

# In[ ]:


sns.scatterplot(data=history_train, x='away_rating_gap', y=history_train.away_goal_difference)


# Let's go further in the relationship between rating and goals scored in the past matches, this time only for the home team. A scatter plot again shows a wide range of possibilities. A team with a rating of 5 can end up scoring 0 to about 7 goals, while a team with a rating of 30 can end up scoring 0 to more than 20 goals. The rating and goals scored of the opponent teams has a similar distribution.

# In[ ]:


sns.jointplot(data=history_train, x='home_team_history_rating', y='home_team_history_goal')


# In[ ]:


sns.jointplot(data=history_train, x='home_team_history_opponent_rating', y='home_team_history_opponent_goal')


# On the surface the wide range of possibilities may make one think the rating only have limited predictive power. But if we look into the 'cross sections', i.e. observing the distributions of goals scored for a given value of rating, it is more apparent that rating makes a difference. As the values of ratings are floats, we can only make observations by binning. In this investigation I use plus and minus 0.5 as the lower and upper limits.

# In[ ]:


r5=history_train.loc[(history_train.home_team_history_rating > 4.5) & (history_train.home_team_history_rating <5.5)].home_team_history_goal
sns.displot(data=r5, discrete=True)


# So teams with a rating of 5 (binned as having ratings between 4.5 and 5.5) mostly score 1 or 0 goal. Teams with a rating of 10 mostly score 1 to 2 goals.

# In[ ]:


r10=history_train.loc[(history_train.home_team_history_rating > 9.5) & (history_train.home_team_history_rating <10.5)].home_team_history_goal
sns.displot(data=r10, discrete=True)


# In[ ]:


r15=history_train.loc[(history_train.home_team_history_rating > 14.5) & (history_train.home_team_history_rating <15.5)].home_team_history_goal
sns.displot(data=r15, discrete=True)


# And teams with a rating of 15 mostly score 2 or 3 goals, teams with a rating of 20 mostly score 3 or 2 goals. The distributions of goals scored look like Poisson distribution, though in the case of 20 it is less well shaped, apparently due to the smaller size of sample.

# In[ ]:


r20=history_train.loc[(history_train.home_team_history_rating > 19.5) & (history_train.home_team_history_rating <20.5)].home_team_history_goal
sns.displot(data=r20, discrete=True)


# If the goals scored form a Poisson distribution, its single parameter lambda can be interpreted as the expected goals scored given a rating. So let's see the mean goals scored for each bin of rating...

# In[ ]:


rating_goal_means=[history_train.loc[(history_train.home_team_history_rating > i-0.5) 
                                     & (history_train.home_team_history_rating < i+0.5)].home_team_history_goal.mean() for i in range (1,21)]


# In[ ]:


px.scatter(x=np.linspace(1,20,20), y=rating_goal_means, trendline="ols", labels={
                     "x": "rating",
                     "y": "mean goals"})


# **The means goals scored nearly fall into a straight line! What's more interesting, we can say in a rough way that each additional 5 points increase in rating results in one more expected goal!!**

# In[ ]:


opponent_goal_means=[history_train.loc[(history_train.home_team_history_opponent_rating > i-0.5) 
                                       & (history_train.home_team_history_opponent_rating < i+0.5)].home_team_history_opponent_goal.mean() 
                     for i in range (1,21)]
px.scatter(x=np.linspace(1,20,20), y=opponent_goal_means, trendline="ols", labels={
                     "x": "rating",
                     "y": "mean goals"})


# The mean goals scored by the opponents also have a similar pattern.

# In[ ]:


home_goal_means=[history_train.loc[(history_train.home_team_history_is_play_home==1) 
                                    & (history_train.home_team_history_rating > i-0.5) 
                                    & (history_train.home_team_history_rating < i+0.5)].home_team_history_goal.mean() 
                  for i in range (1,21)]
away_goal_means=[history_train.loc[(history_train.home_team_history_is_play_home==0) 
                                   & (history_train.home_team_history_rating > i-0.5) 
                                   & (history_train.home_team_history_rating < i+0.5)].home_team_history_goal.mean() 
                 for i in range (1,21)]


# In[ ]:


fig=px.scatter(x=np.linspace(1,20,20), y=home_goal_means, trendline="ols", labels={
                     "x": "rating",
                     "y": "mean goals"})
fig.add_scatter(x=np.linspace(1,20,20), y=away_goal_means, mode="markers",
                marker=dict(size=8, color="Red"),
               name="play away")
fig.show()


# How about home court advantage? We can say the mean goals scored for a team playing home or playing away for a given rating have the same underlying pattern, but the deviances are more notable in higher ratings when playing away, perhaps due to the fact that teams playing away rarely have high ratings, and thus have far smaller samples for such groups. And distributions of ratings show that teams playing home generally do have higher ratings than playing away.

# In[ ]:


play_home_ratings=history_train.loc[history_train.home_team_history_is_play_home==1].home_team_history_rating.dropna()
play_home_ratings.describe()


# In[ ]:


play_away_ratings=history_train.loc[history_train.home_team_history_is_play_home==0].home_team_history_rating.dropna()
play_away_ratings.describe()


# In[ ]:


from plotly.subplots import make_subplots
fig2 = make_subplots(rows=2, subplot_titles=['play home','play away'])
fig2.add_histogram(x=play_home_ratings, row=1, col=1) 
fig2.add_histogram(x=play_away_ratings, row=2, col=1) 
fig2.show()


# In[ ]:




